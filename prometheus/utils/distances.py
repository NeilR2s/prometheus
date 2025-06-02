"""
Distance formulas implemented based on this reference material:
http://www.saedsayad.com/k_nearest_neighbors.htm

This module provides JAX-accelerated functions for calculating common distance
metrics between two vectors. It expects all input values to be array-like
and consist of numerical data. Ensure that any necessary data type casting
or preprocessing is handled before calling these functions.
"""

from functools import partial
import jax.numpy as jnp
import jax
from jax import jit


@jit
def euclidean_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Euclidean distance between two JAX arrays.
    Formula: sqrt(sum((x_i - y_i)^2)).

    Args:
        x: A JAX array representing the first vector (continuous values).
        y: A JAX array representing the second vector (continuous values).
           Must have the same shape as x.

    Returns:
        A JAX scalar array containing the computed Euclidean distance.
    """
    return jnp.sqrt(jnp.sum(jnp.square(x - y)))


@jit
def manhattan_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Manhattan distance between two JAX arrays.
    Formula: sum(|x_i - y_i|).

    Args:
        x: A JAX array representing the first vector (continuous values).
        y: A JAX array representing the second vector (continuous values).
           Must have the same shape as x.

    Returns:
        A JAX scalar array containing the computed Manhattan distance.
    """
    return jnp.sum(jnp.abs(x - y))

@partial(jax.jit, static_argnames=["q"])
def minkowski_distance(x: jnp.ndarray, y: jnp.ndarray, q: int) -> jnp.ndarray:
    """
    Calculates the Minkowski distance between two JAX arrays.
    
    Formula: (sum(|x_i - y_i|^q))^(1/q).
    - q = 1 yields Manhattan distance.
    - q = 2 yields Euclidean distance.

    Args:
        x: A JAX array representing the first vector.
        y: A JAX array representing the second vector. Must have the same shape as x.
        q: The order parameter for the Minkowski distance. Must be > 0.

    Returns:
        A JAX scalar array containing the computed Minkowski distance.

    Exceptions:
        ValueError: If q is not a positive integer.
    """
    if not isinstance(q, int) or q <=0:
        raise ValueError(f"Minkowski distance expects an order q (integer) greater than 0. Referenced value is {q}")
    return jnp.power(jnp.sum(jnp.power(jnp.abs(x - y), q)), 1/q)