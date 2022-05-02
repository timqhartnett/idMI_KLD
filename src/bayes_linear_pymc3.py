#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:09:17 2022

bayesian linear regression using Pymc3 

Note: need to use specific pymc environment as pymc3 causes issues with tensorflow and gpflow on Mac M1
    (hacky work around and should really just resolve the dependancy issues with gpflow and pymc3 but I'm lazy')

@author: timhartnett
"""


import pandas as pd
import pymc3 as pm

working_dir = '/Users/timhartnett/Downloads/3d-int_3-7-21/'

data = pd.read_csv(working_dir+'best_data_1.530.44.csv',index_col=[0])
y = data.loc[:,'idmi'].values
x = data.loc[:,:'idmi'].values

with pm.Model() as model:
    pm.glm.GLM.from_formula('idmi ~ kld + distance',data)
    
    trace = pm.sample()