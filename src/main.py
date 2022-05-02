#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:53:26 2022

@author: timhartnett
"""
#from blr import BayesianLinearRegression
import numpy as np
import pandas as pd
from kld_utits import evaluate_data
from kld_utits import find_dos_similarty
import random
import sys
sys.path.insert(0, '/Users/timhartnett/Downloads/RL_bayesian_optimization/DQN_1')
import bayes_util as bayes
import gpflow

lower = [np.round(random.uniform(0,5),2) for i in range(10)]
upper = [np.round(random.uniform(0,2),2) for i in range(10)]
hyperparameter_dict = {'lower':[],'upper':[],'score':[],'model':[]}
working_dir = '/Users/timhartnett/Downloads/3d-int_3-7-21/'
elements = ['Mn','Fe','Co','Ni','Cr']
#### hyperparameter search for upper and lower energy cuttoffs ####
best_score = 0
for i in range(len(lower)):
    print('#### LOWER = %s ####'%lower[i])
    for j in range(len(upper)):
        print('upper = %s'%upper[j])
        data = find_dos_similarty(upper[i], lower[j], working_dir)
        hyperparameter_dict, score = evaluate_data(data, hyperparameter_dict, upper[j], lower[i])
        if score >= best_score:
            data_best = data
            best_lower = lower[i]
            best_upper = upper[j]
            best_score = score

hyper_df = pd.DataFrame(hyperparameter_dict)
best = hyper_df.sort_values('score',ascending=False).index[0]

### bayesian optimization of hyperparemeters ####

#build virtual dataset 
lower = np.linspace(start=0.01,stop=9,num=900)
upper = np.linspace(start=0.01,stop=2,num=200)
virtual = np.array(np.meshgrid(lower,upper)).T.reshape(-1,2)
for i in range(200):
    x_train = hyper_df.iloc[:,:2].values
    y_train = np.array(hyper_df['score'].values).reshape(-1,1)
    train = np.hstack((x_train,y_train))
    model = bayes.GP_model(train)
    model.train_GP()
    means,var = model.predict_virtual(virtual)
    aqf = bayes.aquisition_functions(means, var, train[:,-1], step_num=i+1)
    
    selection = virtual[aqf.ei(),:]
    upper = selection[1]
    lower = selection[0]
    data = find_dos_similarty(upper, lower, working_dir)
    hyperparameter_dict, score = evaluate_data(data,hyperparameter_dict,upper,lower)
    hyper_df = pd.DataFrame(hyperparameter_dict)
    if score >= best_score:
        data_best = data
        best_lower = lower
        best_upper = upper
        best_score = score

data = data_best
data.to_csv(working_dir+'best_data_'+str(best_lower)+str(best_upper)+'.csv')
noise_std_dev = 0.5
noise_var = noise_std_dev**2
prior_mean = np.array([0, 0, 0])
X = data.loc[:,'kld':'distance'].values
Y = data['idmi'].values


'''
k = gpflow.kernels.Linear(noise_var)
meanf = gpflow.mean_functions.Linear(1.0, 0.0)
m = gpflow.models.GPR(X, Y, k, meanf)
m.likelihood.variance = 0.01


sample_size = 1000





best = hyper_df.sort_values('score',ascending=False).loc[0,:]

model = best['model']
train = data.loc[best['train'],:]
x_train = train.iloc[:,:-1].values
test = data.loc[best['test'],:]
x_test = test.iloc[:,:-1].values
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
'''


