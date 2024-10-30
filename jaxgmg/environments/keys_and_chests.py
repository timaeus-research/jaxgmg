"""
Parameterised environment and level generator for keys and chests problem.
Key components are as follows.

Structs:

* The `Level` struct represents a particular maze layout and key/chest/mouse
  spawn position.
* The `EnvState` struct represents a specific dynamic state of the
  environment.

Classes:

* `Env` class, provides `reset`, `step`, and `render` methods in a
  gymnax-style interface (see `base` module for specifics of the interface).
* `LevelGenerator` class, provides `sample` method for randomly sampling a
  level from a configurable level distribution.
* `LevelParser` class, provides a `parse` and `parse_batch` method for
  designing Level structs based on ASCII depictions.
"""

import enum
import functools
import itertools

from typing import Any

import jax
import jax.numpy as jnp
import einops
from flax import struct
import chex
from jaxtyping import PyTree

from jaxgmg.procgen import maze_generation as mg
from jaxgmg.procgen import maze_solving, combinatorix
from jaxgmg.environments import base


@struct.dataclass
class Level(base.Level):
    """
    Represent a particular environment layout:

    * wall_map : bool[h, w]
            Maze layout (True = wall)
    * keys_pos : index[k, 2]
            List of coordinates of keys (index into `wall_map`)
    * chests_pos : index[c, 2]
            List of coordinates of chests (index into `wall_map`)
    * initial_mouse_pos : index[2]
            Coordinates of initial mouse position (index into `wall_map`)
    * inventory_map : index[k]
            Coordinates of inventory (index into width)
    * hidden_keys : bool[k]
            True for keys that are actually unused within the level.
    * hidden_chests : bool[c]
            True for chests that are actually unused within the level.
    """
    wall_map: chex.Array
    keys_pos: chex.Array
    chests_pos: chex.Array
    initial_mouse_pos: chex.Array
    inventory_map: chex.Array
    hidden_keys: chex.Array
    hidden_chests: chex.Array


@struct.dataclass
class EnvState(base.EnvState):
    """
    Dynamic environment state within a particular level.

    * mouse_pos : index[2]
            Current coordinates of the mouse. Initialised to
            `level.initial_mouse_pos`.
    * got_keys : bool[k]
            Mask tracking which keys have already been collected (True).
            Initially all False.
    * used_keys : bool[k]
            Mask tracking which keys have already been collected and then
            spent to open a chest (True). Initially all False.
    * got_chests : bool[c]
            Mask tracking which chests have already been opened (True).
            Initially all False.
    """
    mouse_pos: chex.Array
    got_keys: jax.Array
    used_keys: jax.Array
    got_chests: jax.Array


@struct.dataclass
class Observation(base.Observation):
    """
    Observation for partially observable Maze environment.

    * image : bool[h, w, c] or float[h, w, rgb]
            The contents of the state. Comes in one of two formats:
            * Boolean: a H by W by C bool array where each channel represents
              the presence of one type of thing (wall, mouse, key in world,
              chest, key in inventory).
            * Pixels: an D.H by D.W by 3 array of RGB float values where each
              D by D tile corresponds to one grid square. (D is level of
              detail.)
    """
    image: chex.Array


class Action(enum.IntEnum):
    """
    The environment has a discrete action space of size 4 with the following
    meanings.
    """
    MOVE_UP     = 0
    MOVE_LEFT   = 1
    MOVE_DOWN   = 2
    MOVE_RIGHT  = 3


class Channel(enum.IntEnum):
    """
    The observations returned by the environment are an `h` by `w` by
    `channel` Boolean array, where the final dimensions 0 through 4 indicate
    the following:

    * `WALL`:   True in the locations where there is a wall.
    * `MOUSE`:  True in the one location the mouse occupies.
    * `KEY`:    True in locations occupied by an uncollected key.
    * `CHEST`:  True in locations occupied by an unopened chest.
    * `INV`:    True in a number of random locations corresponding to the
                number of previously-collected but as-yet-unused keys.
    """
    WALL  = 0
    MOUSE = 1
    KEY   = 2
    CHEST = 3
    INV   = 4


class Env(base.Env):
    """
    Keys and Chests environment.

    In this environment the agent controls a mouse navigating a grid-based
    maze. The mouse must pick up keys located throught the maze and then use
    them to open chests.

    There are four available actions which deterministically move the mouse
    one grid square up, right, down, or left respectively.
    * If the mouse would hit a wall it remains in place.
    * If the mouse hits a key, the key is removed from the grid and stored in
      the mouse's inventory.
    * If the mouse hits a chest and has at least one key in its inventory
      then the mouse opens the chest, reward is delivered, and the key is
      spent. If the mouse doesn't have any keys it passes through the chest.
    """


    @property
    def num_actions(self) -> int:
        return len(Action)

    
    def obs_type( self, level: Level) -> PyTree[jax.ShapeDtypeStruct]:
        # TODO: only works for boolean observations...
        H, W = level.wall_map.shape
        C = len(Channel)
        return Observation(
            image=jax.ShapeDtypeStruct(
                shape=(H, W, C),
                dtype=bool,
            ),
        )


    @functools.partial(jax.jit, static_argnames=('self',))
    def _reset(
        self,
        level: Level,
    ) -> EnvState:
        """
        See reset_to_level method of Underspecified
        """
        num_keys, _2 = level.keys_pos.shape
        num_chests, _2 = level.chests_pos.shape
        state = EnvState(
            mouse_pos=level.initial_mouse_pos,
            got_keys=level.hidden_keys,
            used_keys=level.hidden_keys,
            got_chests=level.hidden_chests,
            level=level,
            steps=0,
            done=False,
        )
        return state
        

    @functools.partial(jax.jit, static_argnames=('self',))
    def _step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
    ) -> tuple[
        EnvState,
        float,
        bool,
        dict,
    ]:
        
        # update mouse position
        steps = jnp.array((
            (-1,  0),   # up
            ( 0, -1),   # left
            (+1,  0),   # down
            ( 0, +1),   # right
        ))
        ahead_pos = state.mouse_pos + steps[action]
        hit_wall = state.level.wall_map[ahead_pos[0], ahead_pos[1]]
        state = state.replace(
            mouse_pos=jax.lax.select(
                hit_wall,
                state.mouse_pos,
                ahead_pos,
            )
        )

        # interact with keys
        pickup_keys = (
            # select keys in the same location as the mouse
            (state.mouse_pos == state.level.keys_pos).all(axis=1)
            # filter for keys the mouse hasn't yet picked up
            & (~state.got_keys)
        )
        state = state.replace(got_keys=state.got_keys ^ pickup_keys)

        # interact with chests
        available_keys = (state.got_keys & ~state.used_keys)
        open_chests = (
            # select chests in the same location as the mouse:
            (state.mouse_pos == state.level.chests_pos).all(axis=1)
            # filter for chests the mouse hasn't yet picked up
            & (~state.got_chests)
            # mask this whole thing by whether the mouse currently has a key
            & available_keys.any()
        )
        the_used_key = jnp.argmax(available_keys) # finds first True if any

        state = state.replace(
            got_chests=state.got_chests | open_chests,
            used_keys=state.used_keys.at[the_used_key].set(
                state.used_keys[the_used_key] | open_chests.any()
            ),
        )
        
        # reward for each chest just opened
        reward = open_chests.sum().astype(float)
        
        # check progress
        # TODO: consider reachability
        available_keys = (~state.level.hidden_keys).sum()
        available_chests = (~state.level.hidden_chests).sum()
        #keys_collected_excess =  ## complete there
        chests_collectable = jnp.minimum(available_keys, available_chests)
        chests_collected = (state.got_chests ^ state.level.hidden_chests).sum()
        done = (chests_collected == chests_collectable)

        # Calculate the number of keys collected (excluding hidden keys)
        keys_collected = (state.got_keys & ~state.level.hidden_keys).sum()

        # Calculate the number of keys used (excluding hidden keys)
        keys_used = (state.used_keys & ~state.level.hidden_keys).sum()

        # Calculate the number of keys currently in inventory
        keys_in_inventory = keys_collected - keys_used

        # Calculate the number of chests remaining to be opened (excluding hidden chests)
        chests_remaining = ((~state.got_chests) & ~state.level.hidden_chests).sum()

        # Calculate the excess keys collected beyond what is needed to open remaining chests

        return (
            state,
            reward,
            done,
            {
                'proxy_rewards': {
                    'keys': keys_in_inventory,
                },
            },
        )


    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_bool(self, state: EnvState) -> Observation:
        """
        Return a boolean grid observation.
        """
        H, W = state.level.wall_map.shape
        C = len(Channel)
        image = jnp.zeros((H, W, C), dtype=bool)

        # render walls
        image = image.at[:, :, Channel.WALL].set(state.level.wall_map)

        # render mouse
        image = image.at[
            state.mouse_pos[0],
            state.mouse_pos[1],
            Channel.MOUSE,
        ].set(True)

        # render keys that haven't been picked up
        image = image.at[
            state.level.keys_pos[:, 0],
            state.level.keys_pos[:, 1],
            Channel.KEY,
        ].set(~state.got_keys)
        
        # render chests that haven't been opened
        image = image.at[
            state.level.chests_pos[:, 0],
            state.level.chests_pos[:, 1],
            Channel.CHEST,
        ].set(~state.got_chests)

        # render keys that have been picked up but haven't been used
        image = image.at[
            0,
            state.level.inventory_map,
            Channel.INV,
        ].set(state.got_keys & ~state.used_keys)

        return Observation(image=image)


    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_obs_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> Observation:
        """
        Return an RGB observation based on a grid of tiles from the given
        spritesheet.
        """
        # get the boolean grid representation of the state
        image_bool = self._render_obs_bool(state).image
        H, W, _C = image_bool.shape

        # find out, for each position, which object to render
        # (for each position pick the first true index top-down this list)
        sprite_priority_vector_grid = jnp.stack([
            # multiple objects
            image_bool[:, :, Channel.MOUSE] & image_bool[:, :, Channel.CHEST],
            image_bool[:, :, Channel.WALL] & image_bool[:, :, Channel.INV],
            # one object
            image_bool[:, :, Channel.WALL],
            image_bool[:, :, Channel.MOUSE],
            image_bool[:, :, Channel.KEY],
            image_bool[:, :, Channel.CHEST],
            # no objects, 'default' (always true)
            jnp.ones((H, W), dtype=bool),
        ])
        chosen_sprites = jnp.argmax(sprite_priority_vector_grid, axis=0)

        # put the corresponding sprite into each square
        spritemap = jnp.stack([
            # multiple objects
            spritesheet['MOUSE_ON_CHEST'],
            spritesheet['KEY_ON_WALL'],
            # one object
            spritesheet['WALL'],
            spritesheet['MOUSE'],
            spritesheet['KEY'],
            spritesheet['CHEST'],
            # no objects
            spritesheet['PATH'],
        ])[chosen_sprites]
        
        image_rgb = einops.rearrange(
            spritemap,
            'h w th tw rgb -> (h th) (w tw) rgb',
        )
        return Observation(image=image_rgb)


    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_state_bool(
        self,
        state: EnvState,
    ) -> chex.Array:
        return self._render_obs_bool(state).image


    @functools.partial(jax.jit, static_argnames=('self',))
    def _render_state_rgb(
        self,
        state: EnvState,
        spritesheet: dict[str, chex.Array],
    ) -> chex.Array:
        return self._render_obs_rgb(state, spritesheet).image


# # # 
# Level generator


@struct.dataclass
class LevelGenerator(base.LevelGenerator):
    """
    Level generator for Keys and Chests environment. Given some maze
    configuration parameters and key/chest sparsity parameters, provides a
    `sample` method that generates a random level.

    * height (int, >= 3):
            the number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width (int, >= 3):
            the number of columns in the grid representing the maze
            (including left and right boundary rows)
    * maze_generator : maze_generation.MazeGenerator
            Provides the maze generation method to use (see module
            `maze_generation` for details).
            The default is a tree maze generator using Kruskal's algorithm.
    * num_keys : int (>= 0)
            the number of keys to randomly place in each generated maze.
    * num_keys_max : int (>0, >= num_keys)
            determines the shape of the key-related arrays in the level
            struct.
    * num_chests : int (>= 0)
            the number of chests to randomly place in each generated maze.
    * num_chests_max : int (>-, >= num_chests)
            determines the shape of the chest-related arrays in the level
            struct.
    """
    height: int = 13
    width: int = 13
    maze_generator : mg.MazeGenerator = mg.TreeMazeGenerator()
    num_keys: int = 1
    num_keys_max: int = 4
    num_chests: int = 4
    num_chests_max: int = 4

    def __post_init__(self):
        assert self.num_keys >= 0
        assert self.num_keys_max > 0
        assert self.num_keys_max >= self.num_keys
        assert self.num_chests >= 0
        assert self.num_chests_max > 0
        assert self.num_chests_max >= self.num_chests
        assert self.num_keys_max <= self.width, "need width for inventory"
        # TODO: somehow prevent or handle too many walls to spawn all items?

    
    @functools.partial(jax.jit, static_argnames=('self',))
    def sample(self, rng: chex.PRNGKey) -> Level:
        """
        Randomly generate a `Level` specification given the parameters
        provided in the constructor of this generator object.
        """
        # construct a random maze
        rng_walls, rng = jax.random.split(rng)
        wall_map = self.maze_generator.generate(
            key=rng_walls,
            height=self.height,
            width=self.width,
        )

        # spawn random mouse pos, keys pos, chests pos
        rng_spawn, rng = jax.random.split(rng)
        coords = einops.rearrange(
            jnp.indices((self.height, self.width)),
            'c h w -> (h w) c',
        )
        all_pos = coords[jax.random.choice(
            key=rng_spawn,
            a=coords.shape[0],
            shape=(1 + self.num_keys_max + self.num_chests_max,),
            p=~wall_map.flatten(),
            replace=False,
        )]
        initial_mouse_pos = all_pos[0]
        keys_pos = all_pos[1:1+self.num_keys_max]
        chests_pos = all_pos[1+self.num_keys_max:]
    
        # decide random positions for keys to display in
        rng_inventory, rng = jax.random.split(rng)
        inventory_map = jax.random.choice(
            key=rng_inventory,
            a=self.width,
            shape=(self.num_keys_max,),
            replace=False,
        )

        # hide keys after the given number
        hidden_keys = (jnp.arange(self.num_keys_max) >= self.num_keys)
        
        # hide chests after the given number
        hidden_chests = (jnp.arange(self.num_chests_max) >= self.num_chests)
    
        return Level(
            wall_map=wall_map,
            initial_mouse_pos=initial_mouse_pos,
            keys_pos=keys_pos,
            chests_pos=chests_pos,
            inventory_map=inventory_map,
            hidden_keys=hidden_keys,
            hidden_chests=hidden_chests,
        )


### Level Mutation
@struct.dataclass
class ToggleWallLevelMutator(base.LevelMutator):

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        # TODO: assuming (h-2)*(w-2) > 2 or something
        
        # which walls are available to toggle?
        valid_map = jnp.ones((h, w), dtype=bool)
        # exclude border
        valid_map = valid_map.at[(0, h-1), :].set(False)
        valid_map = valid_map.at[:, (0, w-1)].set(False)

        # exclude current keys,-chests, mouse spawn positions. 
        for i, chest_pos in enumerate(level.chests_pos):
            valid_map = valid_map.at[chest_pos[0], chest_pos[1]].set(
                jnp.where(
                    level.hidden_chests[i],
                    valid_map[chest_pos[0], chest_pos[1]],
                    False,
                )
            )
        for i, key_pos in enumerate(level.keys_pos):
            valid_map = valid_map.at[key_pos[0], key_pos[1]].set(
                jnp.where(
                    level.hidden_keys[i],
                    valid_map[key_pos[0], key_pos[1]],
                    False,
                )
            )
        valid_map = valid_map.at[
            level.initial_mouse_pos[0],
            level.initial_mouse_pos[1]
        ].set(False)

        valid_mask = valid_map.flatten()

        # pick a random valid position
        coords = einops.rearrange(jnp.indices((h, w)), 'c h w -> (h w) c')
        toggle_pos = jax.random.choice(
            key=rng,
            a=coords,
            axis=0,
            p=valid_mask,
        )

        # toggle the wall there
        hit_wall = level.wall_map[toggle_pos[0], toggle_pos[1]]
        new_wall_map = level.wall_map.at[
            toggle_pos[0],
            toggle_pos[1],
        ].set(~hit_wall)

        return level.replace(wall_map=new_wall_map)


@struct.dataclass
class ScatterMouseLevelMutator(base.LevelMutator):

    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape

        # teleport the mouse to a random location within bounds
        rng_row, rng_col = jax.random.split(rng)
        new_mouse_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_mouse_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_initial_mouse_pos = jnp.array((
            new_mouse_row,
            new_mouse_col,
        ))

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_initial_mouse_pos[0],
            new_initial_mouse_pos[1],
        ].set(False)

        hit_chest = (
            (new_initial_mouse_pos == level.chests_pos).all(axis=1)
        ).any()
        new_chests_pos = level.chests_pos
        new_initial_mouse_pos = jax.lax.select(
            hit_chest,
            level.initial_mouse_pos,
            new_initial_mouse_pos
        )

        hit_key = (
            (new_initial_mouse_pos == level.keys_pos).all(axis=1)
        ).any()
        new_keys_pos = level.keys_pos
        new_initial_mouse_pos = jax.lax.select(
            hit_key,
            level.initial_mouse_pos,
            new_initial_mouse_pos
        )
        
        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            keys_pos=new_keys_pos,
            chests_pos=new_chests_pos,
        )


@struct.dataclass
class ScatterKeyLevelMutator(base.LevelMutator):


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape

        # teleport key to a random location within bounds
        rng_row, rng_col, rng_key = jax.random.split(rng, num=3)
        selected_key = jax.random.choice(
            key = rng_key,
            a = jnp.arange(0, level.keys_pos.shape[0]),
            p = jnp.reshape(~level.hidden_keys, -1),
        )
        new_key_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_key_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_key_pos = jnp.array((
            new_key_row,
            new_key_col,
        ))

        #update keys pos
        new_keys_pos = level.keys_pos.at[selected_key].set(new_key_pos)

        new_initial_mouse_pos = level.initial_mouse_pos


        # check hit chest
        hit_chest = (
           (new_key_pos == level.chests_pos).all(axis=1)
        ).any()

        new_chests_pos = level.chests_pos
        new_keys_pos = jax.lax.select(
            hit_chest,
            level.keys_pos,
            new_keys_pos,
        )

        # Check hit key
        hit_key = (
           (new_key_pos == level.keys_pos).all(axis=1)
        ).any()

        new_keys_pos = jax.lax.select(
            hit_key,
            level.keys_pos,
            new_keys_pos,
        )

        #check hit mouse
        hit_mouse =  (new_key_pos == level.initial_mouse_pos).all()
        new_keys_pos = jax.lax.select(
            hit_mouse,
            level.keys_pos,
            new_keys_pos,
        )
        
        # carve through walls
        new_wall_map = level.wall_map.at[
            new_keys_pos[selected_key][0],
            new_keys_pos[selected_key][1],
        ].set(False)

        # hit_mouse = (level.initial_mouse_pos==new_key_pos).all()

        # new_initial_mouse_pos = jax.lax.select(
        #     hit_mouse,
        #     level.keys_pos[selected_key],
        #     level.initial_mouse_pos,
        # )


        # # upon collision with a chest, swap the key and the chest
        # new_chests_pos = jnp.where((level.chests_pos == new_keys_pos), level.keys_pos[selected_key], level.chests_pos )

        # #upon collision with key, do not do anything
        # new_keys_pos = jnp.where(jnp.all(level.keys_pos == new_key_pos, axis=1)[:, None], 
        #                   level.keys_pos[selected_key], 
        #                   jnp.where(jnp.arange(len(level.keys_pos))[:, None] == selected_key,
        #                           new_key_pos,
        #                           level.keys_pos))
        
        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            keys_pos=new_keys_pos,
            chests_pos=new_chests_pos,

        )


@struct.dataclass
class ScatterChestLevelMutator(base.LevelMutator):


    @functools.partial(jax.jit, static_argnames=["self"])
    def mutate_level(self, rng: chex.PRNGKey, level: Level) -> Level:
        h, w = level.wall_map.shape
        
        # teleport the chest to a random location within bounds
        rng_row, rng_col, rng_chest = jax.random.split(rng, num=3)
        selected_chest = jax.random.choice(
            key = rng_chest,
            a = jnp.arange(0, level.chests_pos.shape[0]),
            p = jnp.reshape(~level.hidden_chests, -1),
        )
        new_chest_row = jax.random.choice(
            key=rng_row,
            a=jnp.arange(1, h-1),
        )
        new_chest_col = jax.random.choice(
            key=rng_col,
            a=jnp.arange(1, w-1),
        )
        new_chest_pos = jnp.array((
            new_chest_row,
            new_chest_col,
        ))

        new_chests_pos = level.chests_pos.at[selected_chest].set(new_chest_pos)

        new_initial_mouse_pos = level.initial_mouse_pos

        # check hit chest
        hit_chest = (
           (new_chest_pos == level.chests_pos).all(axis=1)
        ).any()

        new_chests_pos = jax.lax.select(
            hit_chest,
            level.chests_pos,
            new_chests_pos,
        )

        # Check hit key
        hit_key = (
           (new_chest_pos == level.keys_pos).all(axis=1)
        ).any()

        new_keys_pos = level.keys_pos
        new_chests_pos = jax.lax.select(
            hit_key,
            level.chests_pos,
            new_chests_pos,
        )

        #check hit mouse
        hit_mouse =  (new_chest_pos == level.initial_mouse_pos).all()
        new_chests_pos = jax.lax.select(
            hit_mouse,
            level.chests_pos,
            new_chests_pos,
        )

        # carve through walls
        new_wall_map = level.wall_map.at[
            new_chests_pos[selected_chest][0],
            new_chests_pos[selected_chest][1],
        ].set(False)

        # # carve through walls
        # new_wall_map = level.wall_map.at[
        #     new_chest_pos[0],
        #     new_chest_pos[1],
        # ].set(False)

        # # upon collision with mouse, transpose key with mouse
        # hit_mouse = (new_chest_pos == level.initial_mouse_pos).all()
        # new_initial_mouse_pos = jax.lax.select(
        #     hit_mouse,
        #     level.chests_pos[selected_chest],
        #     level.initial_mouse_pos,
        # )

        # # upon collision with a chest, swap the key and the chest
        # new_keys_pos = jnp.where((level.keys_pos == new_chest_pos), level.chests_pos[selected_chest], level.keys_pos )

        # new_chests_pos = jnp.where(jnp.all(level.chests_pos == new_chest_pos, axis=1)[:, None], 
        #                   level.chests_pos[selected_chest], 
        #                   jnp.where(jnp.arange(len(level.chests_pos))[:, None] == selected_chest,
        #                           new_chest_pos,
        #                           level.chests_pos))

        return level.replace(
            wall_map=new_wall_map,
            initial_mouse_pos=new_initial_mouse_pos,
            keys_pos=new_keys_pos,
            chests_pos = new_chests_pos,

        )

# # # 
# Level parsing


@struct.dataclass
class LevelParser(base.LevelParser):
    """
    Level parser for Keys and Chests environment. Given some parameters
    determining level shape, provides a `parse` method that converts an
    ASCII depiction of a level into a Level struct. Also provides a
    `parse_batch` method that parses a list of level strings into a single
    vectorised Level PyTree object.

    * height (int, >= 3):
            The number of rows in the grid representing the maze
            (including top and bottom boundary rows)
    * width (int, >= 3):
            The number of columns in the grid representing the maze
            (including left and right boundary rows)
    * num_keys_max : int (>0, >= num_keys_min, <= width)
            The largest number of keys that might appear in the level.
            Note: Cannot exceed width as inventory is shown along top row.
    * num_chests_max : int (>-, >= num_chests_min)
            the largest number of chests that might appear in the level.
    * inventory_map : int[num_keys_max] (all are < width)
            The indices into the top row where successive keys are stored.
    * char_map : optional, dict{str: int}
            The keys in this dictionary are the symbols the parser will look
            to define the location of the walls and each of the items. The
            default map is as follows:
            * The character '#' maps to `Channel.WALL`.
            * The character '@' maps to `Channel.MOUSE`.
            * The character 'k' maps to `Channel.KEY`.
            * The character 'c' maps to `Channel.CHEST`.
            * The character '.' maps to `len(Channel)`, i.e. none of the
              above, representing the absence of an item.
    """
    height: int
    width: int
    num_keys_max: int
    num_chests_max: int
    inventory_map: chex.Array
    char_map = {
        '#': Channel.WALL,
        '@': Channel.MOUSE,
        'k': Channel.KEY,
        'c': Channel.CHEST,
        '.': len(Channel), # PATH
    }


    def parse(self, level_str):
        """
        Convert an ASCII string depiction of a level into a Level struct.
        For example:

        >>> p = LevelParser(height=5,width=5,num_keys_max=3,num_chests_max=3)
        >>> p.parse('''
        ... # # # # #
        ... # . k c #
        ... # @ # k #
        ... # k # c #
        ... # # # # #
        ... ''')
        Level(
            wall_map=Array([
                [1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,1,0,1],
                [1,0,1,0,1],
                [1,1,1,1,1],
            ], dtype=bool),
            keys_pos=Array([[1, 2], [2, 3], [3, 1]], dtype=int32),
            chests_pos=Array([[1, 3], [3, 3], [0, 0]], dtype=int32),
            initial_mouse_pos=Array([2, 1], dtype=int32),
            inventory_map=Array([0, 1, 2], dtype=int32),
            hidden_keys=Array([False, False, False], dtype=bool),
            hidden_chests=Array([False, False,  True], dtype=bool),
        )
        """
        # parse into grid of IntEnum elements
        level_grid = [
            [self.char_map[e] for e in line.split()]
            for line in level_str.strip().splitlines()
        ]
        assert len(level_grid) == self.height, "wrong height"
        assert all([len(r) == self.width for r in level_grid]), "wrong width"
        level_map = jnp.asarray(level_grid)
        
        # extract wall map
        wall_map = (level_map == Channel.WALL)
        assert wall_map[0,:].all(), "top border incomplete"
        assert wall_map[:,0].all(), "left border incomplete"
        assert wall_map[-1,:].all(), "bottom border incomplete"
        assert wall_map[:,-1].all(), "right border incomplete"

        # extract key positions and number
        key_map = (level_map == Channel.KEY)
        num_keys = key_map.sum()
        assert num_keys <= self.num_keys_max, "too many keys"
        keys_pos = jnp.stack(
            jnp.where(key_map, size=self.num_keys_max),
            axis=1,
        )
        hidden_keys = (jnp.arange(self.num_keys_max) >= num_keys)
        
        # extract chest positions and number
        chest_map = (level_map == Channel.CHEST)
        num_chests = chest_map.sum()
        assert num_chests <= self.num_chests_max, "too many chests"
        chests_pos = jnp.stack(
            jnp.where(chest_map, size=self.num_chests_max),
            axis=1,
        )
        hidden_chests = (jnp.arange(self.num_chests_max) >= num_chests)

        # extract mouse spawn position
        mouse_spawn_map = (level_map == Channel.MOUSE)
        assert mouse_spawn_map.sum() == 1, "there must be exactly one mouse"
        initial_mouse_pos = jnp.concatenate( # cat for the mouse ;3
            jnp.where(mouse_spawn_map, size=1)
        )

        return Level(
            wall_map=wall_map,
            keys_pos=keys_pos,
            chests_pos=chests_pos,
            initial_mouse_pos=initial_mouse_pos,
            inventory_map=jnp.asarray(self.inventory_map),
            hidden_keys=hidden_keys,
            hidden_chests=hidden_chests,
        )


# # # 
# Level solving (full)


@struct.dataclass
class FullLevelSolution(base.LevelSolution):
    level: Level
    directional_distances: chex.Array


@struct.dataclass
class FullLevelSolver(base.LevelSolver):


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> FullLevelSolution:
        # compute distance between mouse and cheese
        dd = maze_solving.maze_directional_distances(level.wall_map)
        return FullLevelSolution(
            level=level,
            directional_distances=dd,
        )

    
    def _evaluate_visitation_sequence(
        self,
        dists: chex.Array,
        state: EnvState,
        sequence_of_keys: chex.Array,           # int[Combinations(K, N), N], index into dists
        sequence_of_chests: chex.Array,         # int[Combinations(C, N), N], index into dists
        sequence_of_keys_or_chests: chex.Array, # bool[Catalan(N), 2*N]
    ) -> float:
        """
        Simulate a rollout from the current state and compute how much value
        it creates.
        
        Inputs (let K = num keys, C = num chests, N = min(K, C)):

        * dists: float[1+K+C, 1+K+C]
            Distance matrix. The rows columns represent, in order, the mouse,
            key1, ..., keyK, chest1, ..., chestC. The type is 'float' even
            though most distances are integers because some pairs are
            unreachable and for these the distance is stored as floating
            point infinity.
        * state: EnvState
            The state from which the plan to be evaluated is to begin. Used
            for checking which keys/chests have already been collected, for
            example.
            Note that the dists could have been computed from the state, but
            are pre-computed for efficiency reasons. It is assumed that the
            dists correspond to this state.
        * sequence_of_keys: int[N]
            A sequence of key indices, in the range 0, ..., K-1, for indexing
            into state key arrays.
        * sequence_of_chests: int[N]
            A sequence of chest indices, in the range 0, ..., C-1, for
            indexing into state chest arrays.
        * sequence_of_keys_or_chests: bool[2*N]
            A sequence of flags describing how to interleave the key sequence
            and the chest sequence. A value of 'False' indicates to go to the
            next key. A value of 'True' indicates to go to the next chest.
        
        The final three arguments represent a 'visitation sequence' or
        'plan'. Given such a plan and a starting state we can simulate the
        mouse's path through the environment and figure out how much
        discounted reward it would get. This function does that.
        """
        K, _2 = state.level.keys_pos.shape
        
        @struct.dataclass
        class SimulationState:
            # mouse state
            current_node: int           # index into D
            num_keys_in_inv: int
            # visitation sequence state
            keys_visited: int           # index into sequence_of_keys
            chests_visited: int         # index into sequence_of_chests
            # evaluation state
            cumulative_distance: float  # float to handle infinities
            cumulative_reward: float
        
        # initialise simulation state based on current environment state
        initial_simulation_state = SimulationState(
            current_node=0,
            num_keys_in_inv=jnp.sum(state.got_keys & ~state.used_keys),
            keys_visited=0,
            chests_visited=0,
            cumulative_distance=0.0,
            cumulative_reward=0.0,
        )
        
        # simulate one step of the plan/visitation sequence
        def step_simulation(simulation_state, chest_step):
            key_step = ~chest_step

            # where to next?
            next_key = sequence_of_keys[simulation_state.keys_visited]
            next_chest = sequence_of_chests[simulation_state.chests_visited]
            next_node = jnp.where(
                key_step,
                1 + next_key,       # transform for indexing into dists
                1 + K + next_chest, # "
            )
            
            # logic for key step:
            skip_key = (
                key_step
                & state.got_keys[next_key]
                & ~state.level.hidden_keys[next_key]
            )
            get_key = (
                key_step
                & ~skip_key
                & ~state.level.hidden_keys[next_key]
            )

            # logic for chest step:
            skip_chest = (
                chest_step
                & state.got_chests[next_chest]
                & ~state.level.hidden_chests[next_chest]
            )
            hit_chest = (
                chest_step
                & ~skip_chest
                & ~state.level.hidden_chests[next_chest]
            )
            has_key = (simulation_state.num_keys_in_inv > 0)
            open_chest = has_key & hit_chest
                
            # logic for inventory
            new_num_keys_in_inv = (
                simulation_state.num_keys_in_inv
                + get_key       # increment if we picked up a key
                - open_chest    # decrement if we unlocked a chest
            )
            
            # how many maze steps?
            distance = dists[
                simulation_state.current_node,
                next_node,
            ]
            new_cumulative_distance = jnp.where(
                skip_key | skip_chest,
                simulation_state.cumulative_distance,
                simulation_state.cumulative_distance + distance,
            )
            next_node = jnp.where(
                skip_key | skip_chest,
                simulation_state.current_node,
                next_node,
            )

            # compute reward delivered at this step
            raw_reward = open_chest.astype(float)
            # discount reward since it comes in the future
            discount_factor = self.discount_rate ** new_cumulative_distance
            discounted_reward = raw_reward * discount_factor
            # modify reward based on environment configuration
            virtual_time = state.steps + new_cumulative_distance
            penalty_factor = jnp.where(
                self.env.penalize_time,
                # optional linearly decaying penalty factor
                1.0 - .9 * virtual_time / self.env.max_steps_in_episode,
                # or, no penalty
                1.0,
            )
            truncation_factor = jnp.where(
                virtual_time >= self.env.max_steps_in_episode,
                # if the simulation has exceeded allowed time, zero reward
                0.0,
                # else, full reward
                1.0,
            )
            reward = discounted_reward * penalty_factor * truncation_factor

            # update the carry
            new_simulation_state = SimulationState(
                current_node=next_node,
                num_keys_in_inv=new_num_keys_in_inv,
                keys_visited=simulation_state.keys_visited + key_step,
                chests_visited=simulation_state.chests_visited + chest_step,
                cumulative_distance=new_cumulative_distance,
                cumulative_reward=simulation_state.cumulative_reward + reward,
            )
            return new_simulation_state, new_simulation_state

        # scan this function over the steps of the plan
        final_simulation_state, trace = jax.lax.scan(
            step_simulation,
            initial_simulation_state,
            sequence_of_keys_or_chests,
        )
        return final_simulation_state.cumulative_reward


    def _plan(
        self,
        soln: FullLevelSolution,
        state: EnvState,
    ) -> tuple[
        # value
        float,
        # visitation sequence / plan
        tuple[
            chex.Array,
            chex.Array,
            chex.Array,
        ],
    ]:
        """
        Enumerate all 'plausibly-optimal' sequences of key/chest collection,
        evaluate each one, then return the optimal plan and its value.
        """
        K, _2 = state.level.keys_pos.shape
        C, _2 = state.level.chests_pos.shape
        N = min(K, C)

        # compute the abstract distance graph: the distance between the mouse,
        # each key, and each chest
        pos = jnp.concatenate((
            state.mouse_pos[jnp.newaxis],
            state.level.keys_pos,
            state.level.chests_pos,
        ))
        dists = soln.directional_distances[
            pos[:,0],
            pos[:,1],
            pos[:,[0]],
            pos[:,[1]],
            4, # distance from here (rather than after moving in a direction)
        ]

        # enumerate visitation sequences that could plausibly be optimal
        sequences_of_keys = combinatorix.permutations(K, N)
        sequences_of_chests = combinatorix.permutations(C, N)
        sequences_of_which = combinatorix.associations(N).astype(bool)
        # (the cartesian product of these three sets of sequences gives the
        # full set of visitation sequences)

        # vmap the evaluation function over the cartesian triple product of
        # the above arrays, i.e. over all combinations of one sequence of
        # keys, one sequence of chests, and one interleaving sequence.
        ev = functools.partial(
            self._evaluate_visitation_sequence,
            dists,
            state,
        )
        v1 = jax.vmap(ev, in_axes=(None, None, 0)) # :   N,   N, Z 2N ->     Z
        v2 = jax.vmap(v1, in_axes=(None, 0, None)) # :   N, Y N, Z 2N ->   Y Z
        v3 = jax.vmap(v2, in_axes=(0, None, None)) # : X N, Y N, Z 2N -> X Y Z

        # apply the vmapped function to get an array of values
        values = v3(
            sequences_of_keys,
            sequences_of_chests,
            sequences_of_which,
        ) # -> float[X, Y, Z]
        # multidimensional argmax (note: breaks ties by lowest index)
        i, j, k = jnp.unravel_index(
            jnp.argmax(values),
            values.shape,
        )
        
        value = values[i, j, k]
        plan = (
            sequences_of_keys[i],
            sequences_of_chests[j],
            sequences_of_which[k],
        )
        return value, plan


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_value(self, soln: FullLevelSolution, state: EnvState) -> float:
        # find the value of the best plan from this state using the helper
        value, _plan = self._plan(soln, state)
        return value


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action_values(
        self,
        soln: FullLevelSolution,
        state: EnvState,
    ) -> chex.Array: # float[4]
        # TODO: I guess it requires to simulate each action and then evaluate
        # the resulting states separately? this is not necessary for now...!
        raise NotImplementedError("TODO")


    @functools.partial(jax.jit, static_argnames=('self',))
    def state_action(self, soln: FullLevelSolution, state: EnvState) -> int:
        """
        Optimal action from a given state.

        Parameters:

        * soln : FullLevelSolution
                The output of `solve` method for this level.
        * state : EnvState
                The state to compute the optimal action for.
            
        Return:

        * action : int
                An optimal action from the given state.
                
        Notes:

        * If there are multiple equally optimal actions, this method will
          systematically prefer one or another in a complex way depending on
          the implementation. Currently, it finds the first optimal plan
          based on the order the plans happen to be enumerated, and the first
          optimal action for carrying out the first step of that plan
          (according to the order up (0), left (1), down (2), or right (3)).
        * As a special case of this, if there is no achievable value, the
          best plan would be the first plan, which might involve moving the
          mouse towards keys or chests that are unreachable, disabled or have
          already been collected.
        """
        # find an optimal plan from this state using the helper
        _value, plan = self._plan(soln, state)
        sequence_of_keys, sequence_of_chests, sequence_of_keys_or_chests = plan

        # identify the first not-yet-taken step of the plan
        N, = sequence_of_keys.shape
        skip_keys_mask = state.got_keys[sequence_of_keys]
        skip_chests_mask = state.got_chests[sequence_of_chests]
        skip_which_mask = (jnp.zeros(2*N, dtype=bool)
            .at[jnp.where(sequence_of_keys_or_chests, size=N)]
            .set(skip_chests_mask)
            .at[jnp.where(~sequence_of_keys_or_chests, size=N)]
            .set(skip_keys_mask)
        )
        next_key = jnp.argmin(skip_keys_mask) # id of first False (no skip)
        next_chest = jnp.argmin(skip_chests_mask)
        next_which = jnp.argmin(skip_which_mask)

        # identify the target position to execute this step
        target_pos = jnp.where(
            sequence_of_keys_or_chests[next_which],
            state.level.chests_pos[sequence_of_chests[next_chest]],
            state.level.keys_pos[sequence_of_keys[next_key]],
        )

        # choose the action that steps the mouse towards that position
        action = jnp.argmin(soln.directional_distances[
            state.mouse_pos[0],
            state.mouse_pos[1],
            target_pos[0],
            target_pos[1],
            :4, # only consider up/left/down/right, not 'stay' dimension
        ])
        return action


# # # 
# Level solving (level only)


@struct.dataclass
class PartialLevelSolution(base.LevelSolution):
    value: float # just cache the value


@struct.dataclass
class PartialLevelSolver(base.LevelSolver):


    @functools.partial(jax.jit, static_argnames=('self',))
    def solve(self, level: Level) -> PartialLevelSolution:
        return PartialLevelSolution(
            value=self._plan(level),
        )

    
    def _plan(
        self,
        level: Level,
    ) -> float:
        """
        Enumerate all 'plausibly-optimal' sequences of key/chest collection,
        evaluate each one, then return the optimal plan's value.
        """
        K, _2 = level.keys_pos.shape
        C, _2 = level.chests_pos.shape
        N = min(K, C)

        # compute the abstract distance graph: the distance between the mouse,
        # each key, and each chest
        pos = jnp.concatenate((
            level.initial_mouse_pos[jnp.newaxis],
            level.keys_pos,
            level.chests_pos,
        ))
        dists = maze_solving.maze_distances(level.wall_map)[
            pos[:,0],
            pos[:,1],
            pos[:,[0]],
            pos[:,[1]],
        ]

        # enumerate visitation sequences that could plausibly be optimal
        sequences_of_keys = combinatorix.permutations(K, N)
        sequences_of_chests = combinatorix.permutations(C, N)
        sequences_of_which = combinatorix.associations(N).astype(bool)
        # (the cartesian product of these three sets of sequences gives the
        # full set of visitation sequences)

        # vmap the evaluation function over the cartesian triple product of
        # the above arrays, i.e. over all combinations of one sequence of
        # keys, one sequence of chests, and one interleaving sequence.
        ev = functools.partial(
            self._evaluate_visitation_sequence,
            dists,
            level,
        )
        v1 = jax.vmap(ev, in_axes=(None, None, 0)) # :   N,   N, Z 2N ->     Z
        v2 = jax.vmap(v1, in_axes=(None, 0, None)) # :   N, Y N, Z 2N ->   Y Z
        v3 = jax.vmap(v2, in_axes=(0, None, None)) # : X N, Y N, Z 2N -> X Y Z

        # apply the vmapped function to get an array of values
        values = v3(
            sequences_of_keys,
            sequences_of_chests,
            sequences_of_which,
        ) # -> float[X, Y, Z]

        # report the best value available
        return values.max()


    def _evaluate_visitation_sequence(
        self,
        dists: chex.Array,
        level: Level,
        sequence_of_keys: chex.Array,           # int[Combinations(K, N), N], index into dists
        sequence_of_chests: chex.Array,         # int[Combinations(C, N), N], index into dists
        sequence_of_keys_or_chests: chex.Array, # bool[Catalan(N), 2*N]
    ) -> float:
        """
        Simulate a rollout from the initial state of the level and compute
        how much value it creates.
        
        Inputs (let K = num keys, C = num chests, N = min(K, C)):

        * dists: float[1+K+C, 1+K+C]
            Distance matrix. The rows columns represent, in order, the mouse,
            key1, ..., keyK, chest1, ..., chestC. The type is 'float' even
            though most distances are integers because some pairs are
            unreachable and for these the distance is stored as floating
            point infinity.
        * level: Level
            The level for which the plan is to be constructed. Used for
            checking which keys/chests are enabled, for example.
            Note that the dists could have been computed from the level, but
            are pre-computed for efficiency reasons. It is assumed that the
            dists correspond to this level's wall map.
        * sequence_of_keys: int[N]
            A sequence of key indices, in the range 0, ..., K-1, for indexing
            into state key arrays.
        * sequence_of_chests: int[N]
            A sequence of chest indices, in the range 0, ..., C-1, for
            indexing into state chest arrays.
        * sequence_of_keys_or_chests: bool[2*N]
            A sequence of flags describing how to interleave the key sequence
            and the chest sequence. A value of 'False' indicates to go to the
            next key. A value of 'True' indicates to go to the next chest.
        
        The final three arguments represent a 'visitation sequence' or
        'plan'. Given such a plan and a starting state we can simulate the
        mouse's path through the environment and figure out how much
        discounted reward it would get. This function does that.
        """
        K, _2 = level.keys_pos.shape
        
        @struct.dataclass
        class SimulationState:
            # mouse state
            current_node: int           # index into D
            num_keys_in_inv: int
            # visitation sequence state
            keys_visited: int           # index into sequence_of_keys
            chests_visited: int         # index into sequence_of_chests
            # evaluation state
            cumulative_distance: float  # float to handle infinities
            cumulative_reward: float
        
        # initialise simulation state based on current environment state
        initial_simulation_state = SimulationState(
            current_node=0,
            num_keys_in_inv=0,
            keys_visited=0,
            chests_visited=0,
            cumulative_distance=0.0,
            cumulative_reward=0.0,
        )
        
        # simulate one step of the plan/visitation sequence
        def step_simulation(simulation_state, chest_step):
            key_step = ~chest_step

            # where to next?
            next_key = sequence_of_keys[simulation_state.keys_visited]
            next_chest = sequence_of_chests[simulation_state.chests_visited]
            next_node = jnp.where(
                key_step,
                1 + next_key,       # transform for indexing into dists
                1 + K + next_chest, # "
            )
            
            # logic for key step:
            get_key = (
                key_step
                & ~level.hidden_keys[next_key]
            )

            # logic for chest step:
            hit_chest = (
                chest_step
                & ~level.hidden_chests[next_chest]
            )
            has_key = (simulation_state.num_keys_in_inv > 0)
            open_chest = has_key & hit_chest
                
            # logic for inventory
            new_num_keys_in_inv = (
                simulation_state.num_keys_in_inv
                + get_key       # increment if we picked up a key
                - open_chest    # decrement if we unlocked a chest
            )
            
            # how many maze steps?
            distance = dists[
                simulation_state.current_node,
                next_node,
            ]
            new_cumulative_distance = (
                simulation_state.cumulative_distance + distance
            )

            # compute reward delivered at this step
            raw_reward = open_chest.astype(float)
            # discount reward since it comes in the future
            discount_factor = self.discount_rate ** new_cumulative_distance
            discounted_reward = raw_reward * discount_factor
            # modify reward based on environment configuration
            penalty_factor = jnp.where(
                self.env.penalize_time,
                # optional linearly decaying penalty factor
                1.0 - .9 * new_cumulative_distance / self.env.max_steps_in_episode,
                # or, no penalty
                1.0,
            )
            truncation_factor = jnp.where(
                new_cumulative_distance >= self.env.max_steps_in_episode,
                # if the simulation has exceeded allowed time, zero reward
                0.0,
                # else, full reward
                1.0,
            )
            reward = discounted_reward * penalty_factor * truncation_factor

            # update the carry
            new_simulation_state = SimulationState(
                current_node=next_node,
                num_keys_in_inv=new_num_keys_in_inv,
                keys_visited=simulation_state.keys_visited + key_step,
                chests_visited=simulation_state.chests_visited + chest_step,
                cumulative_distance=new_cumulative_distance,
                cumulative_reward=simulation_state.cumulative_reward + reward,
            )
            return new_simulation_state, None

        # scan this function over the steps of the plan
        final_simulation_state, _ = jax.lax.scan(
            step_simulation,
            initial_simulation_state,
            sequence_of_keys_or_chests,
        )
        return final_simulation_state.cumulative_reward


    @functools.partial(jax.jit, static_argnames=('self',))
    def level_value(self, soln: PartialLevelSolution, level: Level) -> float:
        return soln.value


# # # 
# Level valuation


@functools.partial(jax.jit, static_argnames=('env',))
def optimal_value(
    level: Level,
    discount_rate: float,
    env: Env,
) -> float:
    """
    Compute the optimal return from a given level (initial state) for a
    given discount rate. Respects time penalties to the reward and max
    episode length.

    The approach is simplistic---vectorised evaluation of all Dyck paths
    through the abstract key/chest position network, and select the best
    one.
    This will be enough for small numbers of keys and chests, but will not
    scale well in time or memory requirements.

    Parameters:

    * level : Level
            The level to compute the optimal value for. Depends on the
            wall configuration, the initial agent location, and the
            location of each key and chest.
    * discount_rate : float
            The discount rate to apply in the formula for computing
            return.
    * env : Env
            The output also depends on the environment's reward function,
            in turn on:
            * `env.penalize_time` and
            * `env.max_steps_in_episode`.

    Notes:

    * With a steep discount rate or long episodes, this algorithm might
      run into minor numerical issues where small contributions to the
      return from late into the episode are lost.
    * With many keys and chests, the algorithm mighy be slow to compile
      and run and may use a lot of memory. This is because it relies on
      a brute force approach of statically generating and then processing
      a large number of possible key/chest visitation sequences.
    
    TODO:

    * Support computing the return from arbitrary states. This would be
      somewhat harder, requiring I think the following:
      * Use the current mouse position instead of the initial one.
      * Mask out keys and chests that have already been collected.
      * Initialise the simulation using the current number of
        collected/unused keys (rather than 0).
      * Decrease time remaining from max by current number of steps.
    """
    K, _2 = level.keys_pos.shape
    C, _2 = level.chests_pos.shape
    N = min(K, C)

    # compute the abstract distance graph: the distance between the mouse,
    # each key, and each chest
    dist = maze_solving.maze_distances(level.wall_map)
    pos = jnp.concatenate((
        level.initial_mouse_pos[jnp.newaxis],
        level.keys_pos,
        level.chests_pos,
    ))
    D = dist[
        pos[:,0],
        pos[:,1],
        pos[:,[0]],
        pos[:,[1]],
    ]
    # D has the following rows/cols, with three segments:
    #   0    |    1     ...     K    |    K+1           K+C
    # mouse  |  key_1   ...   key_K  |  chest_1  ...  chest_C
    
    # let a 'visitation sequence' be a triple:
    # * some sequence of N keys to visit in order
    # * some sequence of N chests to visit in order
    # * some sequence of balanced parentheses describing how to interleave
    #   the two sequences (encoded as bools, false=key, true=chest)
    # then given such a triple we can simulate the mouse's path through the
    # environment and figure out how much reward it would get

    def eval(seq_keys, seq_chests, seq_key_or_chest):
        @struct.dataclass
        class _Carry:
            # mouse state
            current_node: int           = 0  # index into D, init at spawn
            num_keys_in_inv: int        = 0  # init empty
            # visitation sequence state
            keys_visited: int           = 0  # index into seq_keys
            chests_visited: int         = 0  # index into seq_chests
            # evaluation state
            cumulative_distance: float  = 0. # float to handle infinities
            cumulative_reward: float    = 0.
        
        # simulate one step of the path
        def _step(carry, chest_step):
            key_step = ~chest_step

            # where to next?
            next_node = jnp.where(
                chest_step,
                seq_chests[carry.chests_visited],
                seq_keys[carry.keys_visited],
            )
            
            # how many maze steps
            distance = D[carry.current_node, next_node]
            new_cumulative_distance = carry.cumulative_distance + distance
            
            # logic for key step:
            get_key = key_step & ~level.hidden_keys[next_node-1]

            # logic for chest step:
            has_key = (carry.num_keys_in_inv > 0)
            hit_chest = chest_step & ~level.hidden_chests[next_node-K-1]
            use_key = has_key & hit_chest

            # compute reward with various modified
            raw_reward = use_key.astype(float)
            discount = discount_rate ** new_cumulative_distance
            penalty = (
                1.0
                - env.penalize_time * 0.9 * new_cumulative_distance / env.max_steps_in_episode
            )
            truncation = (new_cumulative_distance < env.max_steps_in_episode)
            reward = raw_reward * discount * penalty * truncation

            # update the carry
            new_carry = _Carry(
                current_node=next_node,
                num_keys_in_inv=carry.num_keys_in_inv + get_key - use_key,
                keys_visited=carry.keys_visited + key_step,
                chests_visited=carry.chests_visited + chest_step,
                cumulative_distance=new_cumulative_distance,
                cumulative_reward=carry.cumulative_reward + reward,
            )
            return new_carry, reward

        # scan _step over seq_key_or_chest
        final_carry, _rewards = jax.lax.scan(
            _step,
            _Carry(),
            seq_key_or_chest,
        )
        return final_carry.cumulative_reward

    # it remains to enumerate visitation sequences that could plausibly be
    # optimal and vmap the evaluation function over them:
    seqs_keys = 1 + combinatorix.permutations(K, N)
    seqs_chests = 1 + K + combinatorix.permutations(C, N)
    seqs_which = combinatorix.associations(N).astype(bool)

    v1 = jax.vmap(eval, in_axes=(None, None, 0))# :   N,   N, k 2N ->     k
    v2 = jax.vmap(v1, in_axes=(None, 0, None))  # :   N, j N, k 2N ->   j k
    v3 = jax.vmap(v2, in_axes=(0, None, None))  # : i N, j N, k 2N -> i j k
    values = vvveval(seqs_keys, seqs_chests, seqs_which) # -> i j k

    return values.max()


@functools.partial(jax.jit, static_argnames=('env',))
def reachable_value(
    level: Level,
    discount_rate: float,
    env: Env,
) -> float:
    # solve the maze
    dists = maze_solving.maze_distances(level.wall_map)
    # count reachable keys
    keys_dists = dists[
        level.initial_mouse_pos[0],
        level.initial_mouse_pos[1],
        level.keys_pos[:, 0],
        level.keys_pos[:, 1],
    ]
    reachable_keys = ~jnp.isinf(keys_dists)
    num_reachable_keys = jnp.sum(reachable_keys & ~level.hidden_keys)
    # count reachable chests
    chests_dists = dists[
        level.initial_mouse_pos[0],
        level.initial_mouse_pos[1],
        level.chests_pos[:, 0],
        level.chests_pos[:, 1],
    ]
    reachable_chests = ~jnp.isinf(chests_dists)
    num_reachable_chests = jnp.sum(reachable_chests & ~level.hidden_chests)
    # max available reward (TODO: this assumes no discounting)
    reachable_value = jnp.minimum(
        num_reachable_keys,
        num_reachable_chests,
    )
    return reachable_value


@functools.partial(jax.jit, static_argnames=('env',))
def original_optimal_value(
    level: Level,
    discount_rate: float,
    env: Env,
) -> float:
    # compute the abstract distance graph: the distance between the mouse,
    # each key, and each chest
    dist = maze_solving.maze_distances(level.wall_map)
    pos = jnp.concatenate((
        level.initial_mouse_pos[jnp.newaxis],
        level.keys_pos,
        level.chests_pos,
    ))
    D = dist[
        pos[:,0],
        pos[:,1],
        pos[:,[0]],
        pos[:,[1]],
    ]

    # (statically) generate a vector of possible visitation sequences
    # * each is a vector of indices into the above distance matrix
    # * the vectors are 1-based because index 0 in the distance matrix
    #   represents the mouse, the implicit start of all visitation
    #   sequences
    num_keys, _2 = level.keys_pos.shape
    num_chests, _2 = level.chests_pos.shape
    visitation_sequences = jnp.array(
        list(itertools.permutations(range(1, 1+num_keys+num_chests))),
        dtype=int,
    )

    # step through each visitation sequence in parallel, computing the
    # return value for that sequence
    num_sequences, _ = visitation_sequences.shape
    initial_carry = (
        jnp.zeros(num_sequences),            # path length so far
        jnp.zeros(num_sequences),            # return so far
        jnp.zeros(num_sequences, dtype=int), # idx of prev visited thing
        jnp.zeros(num_sequences, dtype=int), # inventory num keys so far
    )
    def _step(carry, visit_id):
        length, value, last_visit_id, inventory = carry
        
        # keep track of how many steps
        dist = D[last_visit_id, visit_id]
        new_length = length + dist

        # if we hit a chest, and we have a key, provide reward (1)
        hit_chest = (visit_id >= 1 + num_keys)
        hit_real_chest = (
            hit_chest
            & ~level.hidden_chests[visit_id-1-num_keys]
            # note: invalid index if hit_chest is false, ok because of &
        )
        has_key = (inventory > 0)
        reward = (hit_real_chest & has_key).astype(float)
        # optional time penalty to reward
        penalized_reward = jnp.where(
            env.penalize_time,
            (1.0 - 0.9 * new_length / env.max_steps_in_episode) * reward,
            reward,
        )
        # mask out rewards beyond the end of the episode
        episodes_still_valid = (new_length < env.max_steps_in_episode)
        valid_reward = penalized_reward * episodes_still_valid
        # contribute to return
        discounted_reward = (discount_rate ** new_length) * valid_reward
        new_value = value + discounted_reward

        # if we hit a key, gain a key; if we hit a chest, use a key
        hit_key = (visit_id >= 1) & (visit_id <= num_keys)
        hit_real_key = (
            hit_key
            & ~level.hidden_keys[visit_id-1]
            # note: invalid index if hit_key is false, ok because of &
        )
        new_inventory = (
            inventory
            + hit_real_key
            - (hit_real_chest & has_key)
        )

        new_carry = (
            new_length,
            new_value,
            visit_id,
            new_inventory,
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(
        _step,
        initial_carry,
        visitation_sequences.T,
    )
    
    # the highest return among all of these sequences must be the optimal
    # return
    _, values, _, _ = final_carry
    
    return values.max()


# # # 
# Level complexity metrics


@struct.dataclass
class LevelMetrics(base.LevelMetrics):


    @functools.partial(jax.jit, static_argnames=('self',))
    def compute_metrics(
        self,
        levels: Level,          # Level[num_levels]
        weights: chex.Array,    # float[num_levels]
    ) -> dict[str, Any]:        # metrics
        # basics
        num_levels, h, w = levels.wall_map.shape
        

        def count_reachable_keys_and_chests(level):
            # solve the maze
            dists = maze_solving.maze_distances(level.wall_map)
            # count reachable keys
            keys_dists = dists[
                level.initial_mouse_pos[0],
                level.initial_mouse_pos[1],
                level.keys_pos[:, 0],
                level.keys_pos[:, 1],
            ]
            reachable_keys = ~jnp.isinf(keys_dists)
            num_reachable_keys = jnp.sum(reachable_keys & ~level.hidden_keys)
            # count reachable chests
            chests_dists = dists[
                level.initial_mouse_pos[0],
                level.initial_mouse_pos[1],
                level.chests_pos[:, 0],
                level.chests_pos[:, 1],
            ]
            reachable_chests = ~jnp.isinf(chests_dists)
            num_reachable_chests = jnp.sum(reachable_chests & ~level.hidden_chests)
            return num_reachable_keys, num_reachable_chests
        num_reachable_keys, num_reachable_chests = jax.vmap(count_reachable_keys_and_chests)(levels)
        

        def count_visible_keys_and_chests(level):
            num_visible_keys = jnp.sum(~level.hidden_keys)
            num_visible_chests = jnp.sum(~level.hidden_chests)
            return num_visible_keys, num_visible_chests
        num_visible_keys, num_visible_chests = jax.vmap(count_visible_keys_and_chests)(levels)

        # num walls (excluding border)
        inner_wall_maps = levels.wall_map[:,1:-1,1:-1]
        num_walls = jnp.sum(inner_wall_maps, axis=(1,2))

        # rendered levels in a grid
        def render_level(level):
            state = self.env._reset(level)
            rgb = self.env.render_state(state)
            return rgb
        _, top_64_level_ids = jax.lax.top_k(weights, k=64)
        top_64_levels = jax.tree.map(
            lambda leaf: leaf[top_64_level_ids],
            levels,
        )
        rendered_levels = jax.vmap(render_level)(top_64_levels)
        rendered_levels_pad = jnp.pad(
            rendered_levels,
            pad_width=((0, 0), (0, 1), (0, 1), (0, 0)),
        )
        rendered_levels_grid = einops.rearrange(
            rendered_levels_pad,
            '(level_h level_w) h w c -> (level_h h) (level_w w) c',
            level_w=8,
        )[:-1,:-1] # cut off last pixel of padding

        return {
            'layout': {
                'levels64_img': rendered_levels_grid,
                # number of walls
                'num_walls_avg': num_walls.mean(),
            },
            'counts': {
                'num_visible_keys_avg': num_visible_keys.mean(),
                'num_visible_chests_avg': num_visible_chests.mean(),
                'num_reachable_keys_avg': num_reachable_keys.mean(),
                'num_reachable_chests_avg': num_reachable_chests.mean(),
            },
        }


