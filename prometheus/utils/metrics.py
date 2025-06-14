"""
https://en.wikipedia.org/wiki/Mean_squared_error
https://en.wikipedia.org/wiki/Coefficient_of_determination
https://en.wikipedia.org/wiki/Root_mean_square_deviation
"""

import jax 
import jax.numpy as jnp
from jax import jit

@jit
def mean_squared_error(y_actual, y_pred):
    
    y_actual = jnp.asarray(y_actual)
    y_pred = jnp.asarray(y_pred)

    n_samples = jnp.shape(y_pred)[0]
    mean_squared_error = (1/n_samples) * jnp.sum(jnp.square((y_actual - y_pred)))

    return mean_squared_error

@jit
def r_squared(y_actual, y_pred): # "coefficient of determination"

    y_actual = jnp.asarray(y_actual)
    y_pred = jnp.asarray(y_pred)

    residual_sum_of_squares = jnp.sum(jnp.square(y_actual-y_pred)) 
    total_sum_of_squares = jnp.sum(jnp.square(y_actual - jnp.mean(y_actual)))

    r_squared = 1 - (residual_sum_of_squares/total_sum_of_squares) 
    
    return r_squared


@jit 
def root_mean_squared_error(y_actual, y_pred):

    y_actual = jnp.asarray(y_actual)
    y_pred = jnp.asarray(y_pred)

    n_samples = jnp.shape(y_pred)[0]
    root_mean_squared_error = jnp.sqrt( (1/n_samples) * jnp.sum(jnp.square(y_actual - y_pred)) )

    return root_mean_squared_error


# true_values = [2.5, 3.7, 1.8, 4.0, 5.2]
# predicted_values = [2.1, 3.9, 1.7, 3.8, 5.0]

# print(f"mse: {mean_squared_error(true_values, predicted_values)}")
# print(f"r2: {r_squared(true_values, predicted_values)}")
# print(f"rmse: {root_mean_squared_error(true_values, predicted_values)}")
