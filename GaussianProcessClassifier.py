from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax import jit
import jax.numpy as np
import jax.scipy as scipy

import utils


# Constants needed for approximating logistic sigmoid by error functions
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                  128.12323805, -2010.49422654])[:, np.newaxis]

class GaussianProcessBinaryClassifier:
    """ Binary classification using Gaussian Processes"""

    def __init__(self):
        self.likelihood = lambda x, y: utils.logistic_sigmoid(y*x)
        self.loglikelihood = lambda x, y: np.log(self.likelihood(x, y))
        self.max_iter = 100
        self.posterior_mode = None
        self.covariance_function = utils.squared_exponential
        self.covariance_params = None

    def calculate_posterior_mode(self, train_cov, train_y):
        """Calculate posterior mode using RW Alg 3.1
        Args:
            train_cov: Covariance matrix of training data
            train_y: targets of training data
        """
        # initialize f
        f = np.zeros_like(train_y)
        # create the identity matrix of required shape
        eye = np.eye(f.shape[0])

        previous_log_marginal_likelihood = -1. * np.inf
        K = train_cov
        y = (train_y * 2 - 1)

        # compute the grad and Hessian of likelihood
        grad_likelihood = jax.vmap(jax.grad(self.loglikelihood), in_axes=0, out_axes=0)
        grad_grad_likelihood = jax.vmap(jax.grad(jax.grad(self.loglikelihood)), in_axes=0, out_axes=0)

        for i in range(self.max_iter):
            W = np.diag(-1. * grad_grad_likelihood(f, y))
            sqrt_W = np.sqrt(W)
            B = eye + np.dot(sqrt_W, np.dot(K, sqrt_W))
            # calculate lower triangular Cholesky decomposition
            L = scipy.linalg.cholesky(B, lower=True)
            b = np.dot(W, f) + grad_likelihood(f, y)
            # calculate a using Cholesky solver
            a = b - np.dot(sqrt_W, scipy.linalg.cho_solve((L, True), np.dot(sqrt_W, np.dot(K, b))))
            # update f
            f = np.dot(K, a)

            current_log_marginal_likelihood = -0.5 * np.dot(np.transpose(a), f) + \
                                              np.sum(self.loglikelihood(f, y)) - \
                                              np.sum(np.log(np.diag(L)))

            if current_log_marginal_likelihood - previous_log_marginal_likelihood < 1e-10:
                break
            previous_log_marginal_likelihood = current_log_marginal_likelihood

        self.posterior_mode = f
        return f, current_log_marginal_likelihood

    def predict_probability(self, train_X, train_y, test_X):
        """Predict binary class probability for class 1 based on Alg 3.2 of RW
        Args:
            train_X: the training data array
            train_y: the training targets
            test_X: the test points
        """
        y = (train_y * 2 - 1)
        eye = np.eye(y.shape[0])
        train_cov = utils.kernel_matrix(train_X, train_X, self.covariance_function, self.covariance_params)
        f, _ = self.calculate_posterior_mode(train_cov, train_y)

        # compute the grad and Hessian of likelihood
        grad_likelihood = jax.vmap(jax.grad(self.loglikelihood), in_axes=0, out_axes=0)
        grad_grad_likelihood = jax.vmap(jax.grad(jax.grad(self.loglikelihood)), in_axes=0, out_axes=0)

        # TODO: refactor this into .fit
        W = np.diag(-1. * grad_grad_likelihood(f, y))
        sqrt_W = np.sqrt(W)
        B = eye + np.dot(sqrt_W, np.dot(train_cov, sqrt_W))
        L = scipy.linalg.cholesky(B, lower=True)

        K_star = utils.kernel_matrix(train_X, test_X, self.covariance_function, self.covariance_params)
        f_star = np.dot(np.transpose(K_star), grad_likelihood(f,y))
        v = scipy.linalg.solve(L, np.dot(sqrt_W, K_star))

        K_star_star = utils.kernel_matrix(test_X, test_X, self.covariance_function, self.covariance_params)
        var_f_star = np.diag(K_star_star), - np.einsum("ij,ij->j", v, v)

        # Reference https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/gaussian_process/gpc.py#L305
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = np.sqrt(np.pi / alpha) \
                    * scipy.special.erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS ** 2))) \
                    / (2 * np.sqrt(var_f_star * 2 * np.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + .5 * COEFS.sum()

        return np.vstack((1 - pi_star, pi_star)).T











