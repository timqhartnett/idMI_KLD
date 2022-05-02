# Bayesian Optimization Utilities
import gpflow
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

class GP_model(object):
    '''
    model for bayesian prediction of material properties
    '''
    def __init__(self,training_data):
        self.training_data = training_data

    def train_GP(self,kernel='Matern32',optimizer='L-BFGS-B'):
        X = self.training_data[:,:-1]
        Y = self.training_data[:,-1].reshape(-1,1)
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        k = gpflow.kernels.Matern52()
        self.m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        try:
            self.opt_logs = opt.minimize(self.m.training_loss, self.m.trainable_variables, 
                                         options=dict(maxiter=100))
        except:
            print('train unsuccessful')
    
    def predict_virtual(self,virtual_data):
        X = virtual_data
        X = (X - X.mean()) / X.std()
        norm_mean, norm_var = self.m.predict_y(X)
        mean = norm_mean*self.training_data[:,-1].std()+self.training_data[:,-1].mean()
        var = norm_var*self.training_data[:,-1].std()+self.training_data[:,-1].mean()
        return mean, var

class aquisition_functions():
    def __init__(self, virtual_means, virtual_vars, training_data, step_num):
        self.means = virtual_means
        self.vars = virtual_vars
        self.train = training_data
        self.step_num = step_num

    def random_walk(self):
        new_data_index = np.random.choice(range(len(self.means)),1)
        return new_data_index
    
    def explore(self):
        new_data_index = np.argmax(self.vars)
        return new_data_index

    def exploit(self):
        new_data_index = np.argmax(self.means)
        return new_data_index
    
    def ucb(self,alpha = 0.5,decay_rate=1):
        kappa = 1/(1+self.step_num)*alpha
        ucb = (self.means - max(self.train)) + kappa * self.vars
        new_data_index = np.argmax(ucb)
        return new_data_index
    
    def pi(self,alpha = 0.5,decay_rate=1):
        kappa = 1/(1+decay_rate*self.step_num)*alpha
        gamma_x = (self.means - (max(self.train) + kappa)) / self.vars
        pi = norm.cdf(gamma_x)
        new_data_index = np.argmax(pi)
        return new_data_index

    def ei(self):
        dist = norm
        maximum = np.max(self.train)
        z = (maximum-self.means)/self.vars
        ei = (maximum-self.means)*dist.cdf(z)+(self.vars*dist.pdf(z))
        new_data_index = np.argmax(ei)
        return new_data_index

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
            
            posterior = norm(loc=pred_mean.flatten(), scale=pred_cov**0.5)
            sample = posterior.rvs(size = sample_size)
            posteriors['mean'].append(np.mean(sample))
            posteriors['stdev'].append(np.std(sample))
        
        return posteriors