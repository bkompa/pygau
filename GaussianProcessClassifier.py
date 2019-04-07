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
        self.likelihood = utils.logistic_sigmoid
        self.loglikelihood = lambda x: np.log(self.likelihood(x))
        self.max_iter = 100
        self.posterior_mode = None

    def calculate_posterior_mode(self, train_cov, y):
        """Calculate posterior mode using RW Alg 3.1
        Args:
            train_cov: Covariance matrix of training data
            y: targets of training data
        """
        # initialize f
        f = np.zeros_like(y)
        # create the identity matrix of required shape
        eye = np.eye(f.shape[0])

        previous_log_marginal_likelihood = -1. * np.inf

        for _ in range(self.max_iter):
            K = train_cov
            # compute the Hessian of likelihood
            W = -1. * jax.grad(jax.grad(self.loglikelihood(f)))
            sqrt_W = np.sqrt(W)
            B = eye + np.dot(sqrt_W, np.dot(K, sqrt_W))
            # calculate lower triangular Cholesky decomposition
            L = np.linalg.cholesky(B, lower=True)
            b = np.dot(W, f) + jax.grad(self.loglikelihood(f))
            # calculate a using Cholesky solver
            a = b - np.dot(sqrt_W, scipy.linalg.cho_solve((L, True), np.dot(sqrt_W, np.dot(K, b))))
            # update f
            f = np.dot(K, a)

            current_log_marginal_likelihood = -.5 * np.dot(np.transpose(a), f) + \
                                              self.loglikelihood(f) - \
                                              np.sum(np.diag(L), axis=1)

            if current_log_marginal_likelihood - previous_log_marginal_likelihood < 1e-10:
                break
            previous_log_marginal_likelihood = current_log_marginal_likelihood

        self.posterior_mode = f
        return f, current_log_marginal_likelihood










