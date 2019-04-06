# some basic tests for pygau

import jax.numpy as np
import utils

x1 = np.array([[1., 2.]])
x2 = np.array([[1., 2.]])
x3 = np.array([[1., 3.]])

X = np.vstack((x1, x2, x3))

# test the squared distance function
assert(utils.squared_distance(x1, x2)[0][0] == 0.)
assert(utils.squared_distance(x1, x3)[0][0] == 1.)

# test the squared exponential function
assert(utils.squared_exponential(x1, x2)[0][0] == 1.)
assert(utils.squared_exponential(x1, x3)[0][0] == np.exp(-.5))

# test the kernel function for the same matrix
k_squared_distance = np.array([[0., 0., 1.],
                               [0., 0., 1.],
                               [1., 1., 0.]])
assert(np.equal(utils.kernel_matrix(X, X, utils.squared_distance, None), k_squared_distance).all())

k_se_distance = np.array([[1., 1., np.exp(-.5)],
                          [1., 1., np.exp(-.5)],
                          [np.exp(-.5), np.exp(-.5), 1.]])
assert(np.equal(utils.kernel_matrix(X, X, utils.squared_exponential, None), k_se_distance).all())

# test the kernel function for a matrix of different dimensions
k_001 = np.array([[0., 0., 1.]])
assert(np.equal(utils.kernel_matrix(X, x1, utils.squared_distance, None), k_001.T).all())
assert(np.equal(utils.kernel_matrix(x1, X, utils.squared_distance, None), k_001).all())
