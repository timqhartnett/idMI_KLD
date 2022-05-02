#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:47:57 2022

Bayesian linear regression 2

@author: timhartnett
"""
from scipy.stats import multivariate_normal
from scipy.stats import norm as univariate_normal
import numpy as np
import pandas as pd


working_dir = '/Users/timhartnett/Downloads/3d-int_3-7-21/plots/'

data = pd.read_csv(working_dir+'kld.csv',index_col=([0]))

class BayesianLinearRegression:
    """ Bayesian linear regression
    
    Args:
        prior_mean: Mean values of the prior distribution (m_0)
        prior_cov: Covariance matrix of the prior distribution (S_0)
        noise_var: Variance of the noise distribution
    """
    
    def __init__(self, prior_mean: np.ndarray, prior_cov: np.ndarray, noise_var: float):
        self.prior_mean = prior_mean[:, np.newaxis] # column vector of shape (1, d)
        self.prior_cov = prior_cov # matrix of shape (d, d)
        # We initalize the prior distribution over the parameters using the given mean and covariance matrix
        # In the formulas above this corresponds to m_0 (prior_mean) and S_0 (prior_cov)
        self.prior = multivariate_normal(prior_mean, prior_cov)
        
        # We also know the variance of the noise
        self.noise_var = noise_var # single float value
        self.noise_precision = 1 / noise_var
        
        # Before performing any inference the parameter posterior equals the parameter prior
        self.param_posterior = self.prior
        # Accordingly, the posterior mean and covariance equal the prior mean and variance
        self.post_mean = self.prior_mean # corresponds to m_N in formulas
        self.post_cov = self.prior_cov # corresponds to S_N in formulas
        
    def update_posterior(self, features: np.ndarray, targets: np.ndarray):
        """
        Update the posterior distribution given new features and targets
        
        Args:
            features: numpy array of features
            targets: numpy array of targets
        """
        # Reshape targets to allow correct matrix multiplication
        # Input shape is (N,) but we need (N, 1)
        targets = targets[:, np.newaxis]
        
        # Compute the design matrix, shape (N, 2)
        design_matrix = self.compute_design_matrix(features)

        # Update the covariance matrix, shape (2, 2)
        design_matrix_dot_product = design_matrix.T.dot(design_matrix)
        inv_prior_cov = np.linalg.inv(self.prior_cov)
        self.post_cov = np.linalg.inv(inv_prior_cov +  self.noise_precision * design_matrix_dot_product)
        
        # Update the mean, shape (2, 1)
        self.post_mean = self.post_cov.dot( 
                         inv_prior_cov.dot(self.prior_mean) + 
                         self.noise_precision * design_matrix.T.dot(targets))

        
        # Update the posterior distribution
        self.param_posterior = multivariate_normal(self.post_mean.flatten(), self.post_cov)
                
    def compute_design_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the design matrix. To keep things simple we use simple linear
        regression and add the value phi_0 = 1 to our input data.
        
        Args:
            features: numpy array of features
        Returns:
            design_matrix: numpy array of transformed features
            
        >>> compute_design_matrix(np.array([2, 3]))
        np.array([[1., 2.], [1., 3.])
        """
        n_samples = len(features)
        phi_0 = np.ones(n_samples)
        design_matrix = np.stack((phi_0, features), axis=1)
        return design_matrix
    
 
    def predict(self, features: np.ndarray,sample_size=1000):
        """
        Compute predictive posterior given new datapoint
        
        Args:
            features: 1d numpy array of features
        Returns:
            pred_posterior: predictive posterior distribution
        """
        
        posteriors = {'mean':[],'stdev':[]}
        for feature in features:
            
            design_matrix = self.compute_design_matrix(np.array([feature]))
            
            pred_mean = design_matrix.dot(self.post_mean)
            pred_cov = design_matrix.dot(self.post_cov.dot(design_matrix.T)) + self.noise_var
            
            posterior = univariate_normal(loc=pred_mean.flatten(), scale=pred_cov**0.5)
            sample = posterior.rvs(size = sample_size)
            posteriors['mean'].append(np.mean(sample))
            posteriors['stdev'].append(np.std(sample))
        
        return posteriors

'''
noise_std_dev = 0.5
noise_var = noise_std_dev**2
prior_mean = np.array([0, 0])
prior_cov = 1/2 * np.identity(2)
blr = BayesianLinearRegression(prior_mean, prior_cov, noise_var)
train_features = data['KLD'].values
train_labels = data['idmi'].values
sample_size = 1000

blr.update_posterior(train_features, train_labels)
train_predicted_posterior = [blr.predict(np.array([feature])) for feature in train_features]
'''


