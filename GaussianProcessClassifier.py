from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax import jit
import jax.numpy as np
import jax.scipy as scipy
import jax.ops
import numpy as onp

import utils


# Constants needed for approximating logistic sigmoid by error functions
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                  128.12323805, -2010.49422654])[:, np.newaxis]


class GaussianProcessBinaryClassifier:
    """ Binary classification using Gaussian Processes"""

    def __init__(self):
        self.likelihood = lambda x, y: utils.logistic_sigmoid(x, y)
        self.loglikelihood = lambda x, y: np.log(self.likelihood(x, y))
        self.max_iter = 100
        self.posterior_mode = None
        self.covariance_function = utils.squared_exponential
        self.covariance_explicit = utils.squared_exponential_explicit
        self.covariance_params = None
        self.is_fitted = False

    def fit(self, train_X, train_y):

        self.train_X = onp.copy(train_X)
        self.train_y = onp.copy(train_y)
        self.train_cov = utils.kernel_matrix(self.train_X, self.train_X, self.covariance_function, self.covariance_params)

        if not np.all(np.bitwise_or(self.train_y==1., self.train_y==0.)):
            raise ValueError('Only supports binary 0/1 labels')

        #optimize lml

        self.is_fitted = True



    def calculate_posterior_mode(self, return_intermediates=False):
        """Calculate posterior mode using RW Alg 3.1
        Args:
            self.train_cov: Covariance matrix of training data
            self.train_y: targets of training data
        """
        # initialize f
        f = np.zeros_like(self.train_y)
        # create the identity matrix of required shape
        eye = np.eye(f.shape[0])

        previous_log_marginal_likelihood = -1. * np.inf
        K = self.train_cov
        y = (self.train_y * 2 - 1)

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

            if (current_log_marginal_likelihood - previous_log_marginal_likelihood) < 1e-10:
                break
            previous_log_marginal_likelihood = current_log_marginal_likelihood

        self.posterior_mode = f

        if return_intermediates:
            return f, current_log_marginal_likelihood, (sqrt_W, L, b, a)
        return f, current_log_marginal_likelihood

    def predict_probability(self, test_X):
        """Predict binary class probability for class 1 based on Alg 3.2 of RW
        Args:
            self.train_X: the training data array
            self.train_y: the training targets
            test_X: the test points
        """
        y = self.train_y * 2. - 1.
        eye = np.eye(y.shape[0])
        train_cov = utils.kernel_matrix(self.train_X, self.train_X, self.covariance_function, self.covariance_params)
        f, _ = self.calculate_posterior_mode()

        # compute the grad and Hessian of likelihood
        grad_likelihood = jax.vmap(jax.grad(self.loglikelihood), in_axes=0, out_axes=0)
        grad_grad_likelihood = jax.vmap(jax.grad(jax.grad(self.loglikelihood)), in_axes=0, out_axes=0)

        # TODO: refactor this into .fit
        W = np.diag(-1. * grad_grad_likelihood(f, y))
        sqrt_W = np.sqrt(W)
        B = eye + np.dot(sqrt_W, np.dot(train_cov, sqrt_W))
        L = scipy.linalg.cholesky(B, lower=True)

        K_star = utils.kernel_matrix(self.train_X, test_X, self.covariance_function, self.covariance_params)
        f_star = np.dot(np.transpose(K_star), grad_likelihood(f, y))
        v = scipy.linalg.solve(L, np.dot(sqrt_W, K_star))

        K_star_star = utils.kernel_matrix(test_X, test_X, self.covariance_function, self.covariance_params)
        var_f_star = np.diag(K_star_star) - np.einsum("ij,ij->j", v, v)

        # Reference https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/gaussian_process/gpc.py#L305
        alpha = 1. / (2. * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = np.sqrt(np.pi / alpha) \
                    * scipy.special.erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS ** 2.))) \
                    / (2. * np.sqrt(var_f_star * 2. * np.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

        return np.vstack((1 - pi_star, pi_star)).T

    def log_marginal_likelihood(self, theta=None, return_gradient=False):
        """Compute the log marginal likelihood and/or grad of log marginal likelihood
        Based on RW Algorithm 5.1
        Args:
            self.train_X: the training data
            self.train_y: the training labels in {0,1}
            theta: covariance function parameters
            return_gradient: controls if gradient of log marginal likelihood is returned
        """

        if theta is None and return_gradient is True:
            raise ValueError("Graident calculation requires theta!=None")

        # TODO: calculate the training covariance matrix in the .fit function #can't reuse cause theta might change
        K = utils.kernel_matrix(self.train_X, self.train_X, self.covariance_function, theta)
        y = 2. * self.train_y - 1.
        f, current_log_marginal_likelihood, (sqrt_W, L, _, a) = self.calculate_posterior_mode(True)

        if not return_gradient:
            return current_log_marginal_likelihood

        R = np.dot(sqrt_W, scipy.linalg.cho_solve((L, True), sqrt_W))
        C = scipy.linalg.solve(L, np.dot(sqrt_W, K))

        # TODO: calculate these functions at construction time
        grad_loglikelihood = jax.vmap(jax.grad(self.loglikelihood), in_axes=0, out_axes=0)
        grad3_loglikelihood = jax.vmap(jax.grad(jax.grad(jax.grad(self.loglikelihood))), in_axes=0, out_axes=0)

        s2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C)) * grad3_loglikelihood(f, y)

        d_Z = np.zeros(theta.shape[0])

        # i+2 because arguments 0 and 1 of self.covariance_explicit are data points x and y
        partial_derivatives_of_kernel = [jax.grad(self.covariance_explicit, argnums=(i+2)) for i in range(theta.shape[0])]

        for idx, param in enumerate(theta):
            d_K = utils.kernel_matrix(self.train_X, self.train_X, partial_derivatives_of_kernel[idx], theta, is_grad=True)
            s1 = 0.5 * (np.dot(np.transpose(a), np.dot(d_K, a)) - np.dot(np.ravel(np.transpose(R)), np.ravel(d_K)))
            b = np.dot(d_K, grad_loglikelihood(f, y))
            s3 = b - np.dot(K, np.dot(R, b))
            d_Z = jax.ops.index_update(d_Z, idx, s1 + np.dot(np.transpose(s2), s3))

        return current_log_marginal_likelihood, d_Z
