from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as np
import math


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


def squared_exponential(x1, x2, params):
    if params is None:
        # make all the parameters log 1. = 0.
        return squared_exponential_explicit(x1, x2, 0., 0., 0.)
    return squared_exponential_explicit(x1, x2, *params)


def squared_exponential_explicit(x1, x2, log_amplitude, log_bandwidth, log_length_scale):
    """ The squared exponential function with different scaling parameters
    Args:
        x1: numpy array of data
        x2: numpy array of data
        log_amplitude: the natural log amplitude
        log_bandwidth: the natural log of bandwidtb
        log_length_scale: the natural log of length scale
    Returns:
        The squared exponential kernel value of points x1 and x2
    """

    x1_scaled = x1 / np.exp(log_length_scale)
    x2_scaled = x2 / np.exp(log_length_scale)
    squared_exponential_result = np.exp(log_amplitude) * np.exp(-0.5 * squared_distance(x1_scaled, x2_scaled) /
                                                            np.exp(log_bandwidth))

    return squared_exponential_result


def kernel_matrix(x, y, kernel, kernel_params=None, is_grad=False):
    """The kernel matrix for arrays of data x and y
    Args:
        x: a data matrix with points stacked in dim 0
        y: a data matrix with points stacked in dim 0
        kernel: a function such that k[i,j] = kernel(x[i], y[j], kernel_params)
        kernel_params: the parameters for the corresponding kernel function
        is_grad: determines whether to unpack parameters for jax grad function or not
    Returns: The 2D kernel matrix k[i,j] = kernel(kernel_params, x[i], y[j])
    """
    if is_grad:
        return jax.vmap(lambda x_i: jax.vmap(lambda y_j: kernel(x_i, y_j, *kernel_params))(y))(x)
    return jax.vmap(lambda x_i: jax.vmap(lambda y_j: kernel(x_i, y_j, kernel_params))(y))(x)



def logistic_sigmoid(x, y):
    """The logistic sigmoid function"""
    return 1. / (1. + np.exp(-1. * y * x))



