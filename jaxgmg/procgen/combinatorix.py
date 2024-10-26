"""
Counting and enumerating combinatorics things in JAX.
"""

import math
import itertools
import functools
import jax
import jax.numpy as jnp

from jaxtyping import Int, Array
from collections.abc import Generator


def num_combinations(n: int, r: int) -> int:
    # meant to be static, uses pure python
    return math.comb(n, r)


def num_permutations(n: int, r: int) -> int:
    # meant to be static, uses pure python
    return math.perm(n, r)


def num_associations(n: int) -> int:
    # meant to be static, uses pure python
    return math.comb(2*n, n) // (n+1)


@functools.partial(jax.jit, static_argnames=["n", "r"])
def combinations(n: int, r: int) -> Int[Array, "num_combinations(n,r) r"]:
    # NOTE: Faster approach may exist, we don't need lex order.
    # Could try more_itertools.distinct_combinations?
    combinations = itertools.combinations(range(n), r)
    return jnp.array(list(combinations))


@functools.partial(jax.jit, static_argnames=["n", "r"])
def permutations(n: int, r: int) -> Int[Array, "num_permutations(n,r) r"]:
    # NOTE: Faster approach may exist, we don't need lex order.
    # Could try more_itertools.distinct_permutations?
    permutations = itertools.permutations(range(n), r)
    return jnp.array(list(permutations))


@functools.partial(jax.jit, static_argnames=["n"])
def associations(n: int) -> Int[Array, "num_associations(n) 2*n"]:
    associations = enumerate_associations(n)
    return jnp.array(list(associations))
    

def enumerate_associations(n: int) -> list[tuple[int, ...]]:
    """
    Returns a list of 'associations' of length `n`, which is my name for
    sequences of `n` pairs of balanced parantheses (or many other things
    counted by the Catalan numbers).

    Actually these sequences are tuples of `n` zeros and `n` ones rather than
    `n` pairs of parentheses. The tuples satisfy the property that no prefix
    of the tuple has more ones than zeros. For example:

    ```
    >>> enumerate_associations(2)
    [
        (0, 1, 0, 1),
        (0, 0, 1, 1),
    ]
    >>> enumerate_associations(3)
    [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 0, 1, 1),
        (0, 0, 1, 1, 0, 1),
        (0, 0, 1, 0, 1, 1),
        (0, 0, 0, 1, 1, 1),
    ]
    
    ```

    The enumeration method is inspired by the following unambiguous grammar
    for the language of balanced parentheses:

                                S  ->  ε
                                S  ->  ( S ) S

    The variables 'lhs' and 'rhs' in the code correspond to the things you
    could substitute for the first and second (resp.) occurrence of the
    variable S in the second rule.

    It also traces out the way you would implement this recursive formula for
    the Catalan numbers if you were counting the balances parentheses:

        C(0) = 1
        C(k) = Sum from j=0 to k-1 of: C(j) * C(k-1-j)      for k > 0

    The first loop is to build up the recurrence from k=1 to n. The second
    loop is the sum over j. The third and fourth loops implement the product
    C(j) * C(k-1-j).

    Caveats:

    * The resulting sequences are not produced in lexicographic order, nor
      reverse lexicographic order. However, the first sequence should always
      be `(0,1,)*n` and the last should always be `(0,)*n + (1,)*n`.

    * The use of tuple concatenation could be a bottleneck for deep recurrence,
      but seems kinda unavoidable and kinda unlikely to become a problem in
      practice since the number of associations grows so rapidly in depth.
    """
    # dynamic programming table
    associations_of_length = {}
    # base case
    associations_of_length[0] = [()]
    # recursive case
    for k in range(1, n+1): # 1 to n inclusive
        # sum
        associations_of_length[k] = []
        for j in range(k):
            # product
            for lhs in associations_of_length[j]:
                for rhs in associations_of_length[k-1-j]:
                    associations_of_length[k].append((0,) + lhs + (1,) + rhs)
    return associations_of_length[n]

