from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as np
import jax.scipy as scipy

import utils


class GaussianProcessBinaryClassifier:
    """ Binary classification using Gaussian Processes"""

    def __init__(self):
        self.likelihood = lambda x, y: utils.logistic_sigmoid(y*x)
        self.loglikelihood = lambda x, y: np.log(self.likelihood(x, y))
        self.max_iter = 100
        self.posterior_mode = None

    def calculate_posterior_mode(self, train_cov, train_y):
        """Calculate posterior mode using RW Alg 3.1
        Args:
            train_cov: Covariance matrix of training data
            y: targets of training data
        """
        # initialize f
        f = np.zeros_like(train_y)
        # create the identity matrix of required shape
        eye = np.eye(f.shape[0])

        previous_log_marginal_likelihood = -1. * np.inf
        K = train_cov
        y = (train_y * 2 - 1)
        print('K: {}'.format(K))

        # map loglikelihood to make y and f and map grad function to be wrt f
        for i in range(self.max_iter):
            print('Iteration {}'.format(i))
            # compute the Hessian of likelihood
            grad_grad_Psi = jax.vmap(jax.grad(jax.grad(self.loglikelihood)), in_axes=0, out_axes=0)
            W = np.diag(-1. * grad_grad_Psi(f, y))
            print('W: {}'.format(W))
            sqrt_W = np.sqrt(W)
            print('W1/2: {}'.format(sqrt_W))
            B = eye + np.dot(sqrt_W, np.dot(K, sqrt_W))
            print('B: {}'.format(B))
            # calculate lower triangular Cholesky decomposition
            L = scipy.linalg.cholesky(B, lower=True)
            print('L: {}'.format(L))
            grad_Psi = jax.vmap(jax.grad(self.loglikelihood), in_axes=0, out_axes=0)
            print('grad_psi: {}'.format(grad_Psi(f, y)))
            print('Wf {}'.format(np.dot(W,f)))
            b = np.dot(W, f) + grad_Psi(f, y)
            print('b: {}'.format(b))
            # calculate a using Cholesky solver
            a = b - np.dot(sqrt_W, scipy.linalg.cho_solve((L, True), np.dot(sqrt_W, np.dot(K, b))))
            print('a: {}'.format(a))
            # update f
            f = np.dot(K, a)
            print('f: {}'.format(f))

            current_log_marginal_likelihood = -0.5 * np.dot(np.transpose(a), f) + \
                                              np.sum(self.loglikelihood(f, y)) - \
                                              np.sum(np.log(np.diag(L)))

            print(current_log_marginal_likelihood)
            if current_log_marginal_likelihood - previous_log_marginal_likelihood < 1e-10:
                break
            previous_log_marginal_likelihood = current_log_marginal_likelihood

        self.posterior_mode = f
        return f, current_log_marginal_likelihood








