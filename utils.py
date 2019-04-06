from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as np


def squared_distance(x1, x2, params=None):
    """ Squared distance function between two numpy arrays
    Args:
        x1: a numpy array
        x2: a numpy array
        params: unused arg so squared distance can be a kernel function
    Returns:
        The L2 distance
    """
    if params is not None:
        raise UserWarning("params is expected to be None for squared_distance")

    return -2. * np.dot(x1, x2.T) + np.sum(x2**2, axis=1) + np.sum(x1**2, axis=1)


def squared_exponential(x1, x2, params=None):
    """ The squared exponential function with different scaling parameters
    Args:
        params: array of length scale parameters
            params[0] log amplitude
            params[1] log bandwidth
            params[2] log length-scale
        x1: numpy array of data
        x2: numpy array of data
    Returns:
        The squared exponential kernel value of points x1 and x2
    """
    if params is None:
        # make all the parameters log 1. = 0.
        params = np.array([0., 0., 0.])

    x1_scaled = x1 / np.exp(params[1])
    x2_scaled = x2 / np.exp(params[2])
    squared_exponential_result = np.exp(params[0]) * np.exp(-0.5 * squared_distance(x1_scaled, x2_scaled) /
                                                            np.exp(params[2]))

    return squared_exponential_result


def kernel_matrix(x, y, kernel, kernel_params=None):
    """The kernel matrix for arrays of data x and y
    Args:
        x: a data matrix with points stacked in dim 0
        y: a data matrix with points stacked in dim 0
        kernel: a function such that k[i,j] = kernel(x[i], y[j], kernel_params)
        kernel_params: the parameters for the corresponding kernel function
    Returns: The 2D kernel matrix k[i,j] = kernel(kernel_params, x[i], y[j])
    """
    return jax.vmap(lambda x_i: jax.vmap(lambda y_j: kernel(x_i, y_j, kernel_params))(y))(x)






