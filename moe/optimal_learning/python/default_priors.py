'''
Created on Oct 14, 2015

@author: Aaron Klein
'''
import numpy as np

from moe.optimal_learning.python.base_prior import BasePrior, TophatPrior, \
    LognormalPrior, HorseshoePrior, NormalPrior


class DefaultPrior(BasePrior):

    def __init__(self, n_dims, dim, num_noise):
        # The number of hyperparameters
        self.n_dims = n_dims
        self.dim = dim
        # The number of noises
        self.num_noise = num_noise
        # Prior for the Matern52 lengthscales
        self.tophat = TophatPrior(-5, 2)

        # Prior for the covariance amplitude
        self.ln_prior = NormalPrior(mean=0.0, sigma=1.0)

        # Prior for the noise
        self.horseshoe = HorseshoePrior(scale=0.1)

    def lnprob(self, theta):
        lp = 0
        # Covariance amplitude
        for i in xrange(self.dim):
            lp += self.ln_prior.lnprob(theta[i])
        for i in xrange(2*self.dim, self.n_dims-self.num_noise):
            lp += self.ln_prior.lnprob(theta[i])
        # Lengthscales
        lp += self.tophat.lnprob(theta[self.dim:2*self.dim])
        # Noise
        for i in xrange(self.num_noise, 0, -1):
            lp += self.horseshoe.lnprob(theta[-i])
        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, self.n_dims])
        # Covariance amplitude
        cs_sample = np.array([self.ln_prior.sample_from_prior(n_samples)[:, 0]
                       for _ in xrange(self.dim)]).T
        p0[:, :self.dim] = cs_sample
        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in xrange(self.dim, 2*self.dim)]).T
        p0[:, self.dim:2*self.dim] = ls_sample
        # Covariance amplitude
        cs_sample = np.array([self.ln_prior.sample_from_prior(n_samples)[:, 0]
                              for _ in xrange(2*self.dim, self.n_dims-self.num_noise)]).T
        p0[:, 2*self.dim:-self.num_noise] = cs_sample
        # Noise
        ns_sample = np.array([self.horseshoe.sample_from_prior(n_samples)[:, 0]
                              for _ in xrange(self.num_noise)]).T
        p0[:, -self.num_noise:] = ns_sample

        return p0
