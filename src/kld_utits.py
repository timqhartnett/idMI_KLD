#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:58:59 2022

@author: timhartnett
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from sklearn import linear_model

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

#filename = '/Users/timhartnett/Downloads/3d-int_3-7-21/Mn_DOS/'
#data = np.loadtxt(fname=filename)
elements = ['Mn','Fe','Co','Ni','Cr']
idmi = np.array([9.18,6.94,5.15,3.58,1.57])



working_dir = '/Users/timhartnett/Downloads/3d-int_3-7-21'

class dos_similarity:
    def __init__(self,working_directory):
        self.working_directory=working_directory
        
    def simple_difference(self,p,q,stepsize):
        return np.sum(abs(p-q)*stepsize)

    def kl_divergence(self,p, q):
        return abs(np.sum(np.where(p != 0, p * np.log(p / q), 0)))
        
    def atom_pdos(self,pdos_data,atom_number,orbital_name,Ef,pdos_directory):
        filename = glob.glob(pdos_directory+'*atm#%s*'%atom_number+orbital_name+'*')
        array = np.loadtxt(filename[0])
        array[:,0] = array[:,0]-Ef
        array[:,2::2] = array[:,2::2]*-1
        pdos_data['atom_%s'%atom_number] = array
        return pdos_data
        
        
    def gather_pdos(self,nscf_directory,pdos_directory,atoms):
        nscf_filename = glob.glob(nscf_directory+'*nscf.out*')
        pdos_data={}
        
        with open(nscf_filename[0]) as nscf_file:
            lines = nscf_file.readlines()
            Fermi_line = [line for line in lines if 'the Fermi energy is' in line]
            Ef = float(Fermi_line[0].split()[-2])
            
        for atom_number in atoms:
            pdos_data = self.atom_pdos(pdos_data,atom_number,'d',Ef,pdos_directory)
        
        return pdos_data
    
    def calculate_tot_KLD(self,pdos_data,lower_bound,upper_bound,mag_nums,cap_nums):
        energy_values = pdos_data['atom_%s'%mag_nums[0]][:,0]
        x = abs(energy_values - upper_bound)
        high = np.argmin(x)
        x = abs(energy_values + lower_bound)
        low = np.argmin(x)
        mag_up = np.zeros(pdos_data['atom_%s'%mag_nums[0]][low:high,0].shape)
        mag_down = np.zeros(pdos_data['atom_%s'%mag_nums[0]][low:high,0].shape)
        cap_up = np.zeros(pdos_data['atom_%s'%mag_nums[0]][low:high,0].shape)
        cap_down = np.zeros(pdos_data['atom_%s'%mag_nums[0]][low:high,0].shape)
        
        for i in range(len(mag_nums)):
            mag_up = mag_up + pdos_data['atom_%s'%mag_nums[i]][low:high,1]
            mag_down = mag_down + pdos_data['atom_%s'%mag_nums[i]][low:high,2]
            cap_up = cap_up + pdos_data['atom_%s'%cap_nums[i]][low:high,1]
            cap_down = cap_down + pdos_data['atom_%s'%cap_nums[i]][low:high,2]
        
        tot_kld = self.kl_divergence(mag_up,cap_up)+self.kl_divergence(mag_down,cap_down)
        
        return tot_kld
    
    def find_moment(self,atom_number,pdos_directory):
        filename = glob.glob(pdos_directory+'pdos.out')
        
        with open(filename[0]) as pdos_file:
            lines = pdos_file.readlines()
            
        atom_index = [c for c,line in enumerate(lines)\
                      if '  Atom #  '+str(atom_number)+': total charge =' in line]
        polarization_line = lines[atom_index[0]+9]
        moment = float(polarization_line.split()[2].split(',')[0])
        return moment
    
    def compute_distance(self,a,b):
        dist = np.sqrt(np.sum((b-a)**2))
        return dist
            
    
    def get_interaction_lengths(self,nscf_directory,mag_numbers,cap_numbers):
        nscf_filename = glob.glob(nscf_directory+'*nscf.in*')
        with open(nscf_filename[0]) as nscf_file:
            lines = nscf_file.readlines()
        atoms = [line for line in lines if 'nat =' in line]
        n_atoms = int(atoms[0].split()[2].split(',')[0])
        start_cell = [i for i,line in enumerate(lines) if 'CELL_PARAMETERS' in line]
        start_atomic = [i for i,line in enumerate(lines) if 'ATOMIC_POSITIONS' in line]
        cell = [line for line in lines[start_cell[0]+1:start_cell[0]+4]]
        cell_array = []
        for line in cell:
            numbers = line.split()
            cell_array.append([float(num) for num in numbers])
        cell_array = np.array(cell_array)
        positions = [line for line in lines[start_atomic[0]+1:start_atomic[0]+n_atoms+1]]
        position_array = []
        for line in positions:
            numbers = line.split()[1:4]
            position_array.append([float(num) for num in numbers])
        position_array = np.array(position_array)
        
        structure = np.dot(position_array,cell_array)
        distances = []
        for i in range(len(mag_numbers)):
            distances.append(self.compute_distance(structure[mag_numbers[i]-1,:],
                                                   structure[cap_numbers[i]-1,:]))
        return distances
    
def train_test(df,split = 0.7):
    train = df.sample(n=int(np.floor(df.shape[0]*split)))
    test = df.drop(list(train.index),axis=0)
    return train, test

def find_dos_similarty(upper,lower,working_dir):
    
    ### Pt substrate ###
    atoms = [str(i) for i in np.arange(21,29)]
    cap_nums = atoms[:4]
    mag_nums = atoms[4:]
    lower_bound = lower
    upper_bound = upper
    tot_kld = []
    tot_distance = []
    
    for element in elements:
        Pt_dos_similarity = dos_similarity(working_dir)
        dos_directory = working_dir+'data/DOS/' + element +'_DOS/'
        pdos_directory = dos_directory + 'PDOS/'
        nscf_directory = dos_directory
        pdos_data = Pt_dos_similarity.gather_pdos(dos_directory, pdos_directory, atoms)
        
        tot_kld.append(Pt_dos_similarity.calculate_tot_KLD(pdos_data, lower_bound, upper_bound, mag_nums, cap_nums))
        mag_nums = [int(num) for num in mag_nums]
        cap_nums = [int(num) for num in cap_nums]
        tot_distance.append(np.mean(Pt_dos_similarity.get_interaction_lengths(nscf_directory,mag_nums,cap_nums)))
    
    idmi = np.array([9.18,6.94,5.15,3.58,1.57])
    Pt_df = pd.DataFrame({'kld':np.array(tot_kld)/4,'distance':tot_distance,'idmi':idmi})

    
    ### W substrate ###
    atoms = [str(i) for i in np.arange(13,17)]
    cap_nums = atoms[:2]
    mag_nums = atoms[2:]
    lower_bound = lower
    upper_bound = upper
    tot_kld = []
    tot_distance = []
    
    for element in elements:
        W_dos_similarity = dos_similarity(working_dir)
        dos_directory = working_dir + 'data/DOS/' +element +'-W001/dos/'
        pdos_directory = dos_directory + 'pdos/'
        nscf_directory = dos_directory
        pdos_data = W_dos_similarity.gather_pdos(dos_directory, pdos_directory, atoms)
        
        tot_kld.append(W_dos_similarity.calculate_tot_KLD(pdos_data, lower_bound, upper_bound, mag_nums, cap_nums))
        mag_nums = [int(num) for num in mag_nums]
        cap_nums = [int(num) for num in cap_nums]
        tot_distance.append(np.mean(W_dos_similarity.get_interaction_lengths(nscf_directory,mag_nums,cap_nums)))
    
    idmi = np.array([17.06,4.35,1.35,0.81,11.68])
    W_df = pd.DataFrame({'kld':np.array(tot_kld)/2,'distance':tot_distance,'idmi':idmi})
    data = pd.concat([Pt_df,W_df],axis=0,ignore_index=True)
    return data

def evaluate_data(data,hyperparameter_dict,upper,lower):
    
    hyperparameter_dict['lower'].append(lower)
    hyperparameter_dict['upper'].append(upper)
    scores = []
    for i in range(data.shape[0]*2):
        train, test = train_test(data)
        X = train.iloc[:,:-1].values
        y = train['idmi'].values
        X_test = test.iloc[:,:-1].values
        y_test = test['idmi'].values
        ols=linear_model.LinearRegression()
        ols.fit(X,y)
        scores.append(ols.score(X_test, y_test))
        
    median_score = np.median(scores)
    
    hyperparameter_dict['score'].append(median_score)
    hyperparameter_dict['model'].append(ols)
    return hyperparameter_dict,median_score

    
'''
x = W_dos_similarity.get_interaction_lengths(nscf_directory,mag_nums,cap_nums)

data = pd.read_csv(working_dir+'plots/kld.csv',index_col=([0]))
data['KLD'] = data['KLD']/2
noise_std_dev = 2
noise_var = noise_std_dev**2
prior_mean = np.array([0, 0])
prior_cov = 1/2 * np.identity(2)
blr = BayesianLinearRegression(prior_mean, prior_cov, noise_var)
train_features = data['KLD'].values/2
train_labels = data['idmi'].values
sample_size = 1000
test_features = np.array(tot_kld)
blr.update_posterior(train_features, train_labels)
train_predicted_posterior = blr.predict(features = train_features,sample_size=1000)
test_predicted_posterior = blr.predict(features = test_features,sample_size=1000)
test_true = np.array([17.06,4.35,1.35,0.81,11.68])
W_df = pd.DataFrame({'idmi':test_true,'KLD':tot_kld})
new_data = pd.concat([data,W_df],axis=0,ignore_index=True)

### sample from both

train = new_data.sample(n=7)
test = new_data.drop(train.index)
train_features = np.exp(-train['KLD'].values)
#train_labels = train['idmi'].values

blr.update_posterior(train_features, train_labels)
train_predicted_posterior = blr.predict(features = train_features,sample_size=1000)
train_predicted_posterior = blr.predict(features = train_features,sample_size=1000)
test_predicted_posterior = blr.predict(features = test_features,sample_size=1000)
'''

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