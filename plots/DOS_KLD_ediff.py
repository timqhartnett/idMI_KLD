#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:54:27 2022

3d-int DOS plots

@author: timhartnett
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from blr import BayesianLinearRegression as blr

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

#filename = '/Users/timhartnett/Downloads/3d-int_3-7-21/Mn_DOS/'
#data = np.loadtxt(fname=filename)
elements = ['Mn','Fe','Co','Ni','Cr']
idmi = np.array([9.18,6.94,5.15,3.58,1.57])

def simple_difference(p,q):
    return np.sum(abs(p-q))

def kl_divergence(p, q):
    return abs(np.sum(np.where(p != 0, p * np.log(p / q), 0)))


KLD_correlations = np.zeros([10,10])
diff_correlations = np.zeros([10,10])
tot_KLD= []
tot_DIFF= []
for element in elements:
    dos_directory  = '/Users/timhartnett/Downloads/3d-int_3-7-21/'+element+'_DOS/'
    nscf_filename = glob.glob(dos_directory+'*nscf.out*')
    with open(nscf_filename[0]) as nscf_file:
        lines = nscf_file.readlines()
        Fermi_line = [line for line in lines if 'the Fermi energy is' in line]
        E_f = float(Fermi_line[0].split()[-2])
        
    pdos_directory = dos_directory+'PDOS'
    Pt = list(np.arange(21,25))
    Mn = list(np.arange(25,29))
    pdos_data = {}
    
    for c,atom in enumerate(Mn):
        filename = glob.glob(pdos_directory+'/*atm#'+str(atom)+'*(d)*')
        array = np.loadtxt(filename[0])
        array[:,0] = array[:,0]-E_f
        array[:,2::2] = array[:,2::2]*-1
        pdos_data['Mn'+str(c)] = array
    
    for c,atom in enumerate(Pt):
        filename = glob.glob(pdos_directory+'/*atm#'+str(atom)+'*(d)*')
        array = np.loadtxt(filename[0])
        array[:,0] = array[:,0]-E_f
        array[:,2::2] = array[:,2::2]*-1
        pdos_data['Pt'+str(c)] = array
    
    energy_values = array[:,0]
    x = abs(energy_values - 2)
    high = np.argmin(x)
    x = abs(energy_values + 1.5)
    low = np.argmin(x)
    
    KLD = {}
    DIFF = {}
    fig = plt.figure(figsize = (10,6))
    ax = plt.axes((0.1,0.1,0.8,0.8))
    Mn_up = np.zeros(pdos_data['Mn1'][low:high,0].shape)
    Mn_down = np.zeros(pdos_data['Mn1'][low:high,0].shape)
    Pt_up = np.zeros(pdos_data['Mn1'][low:high,0].shape)
    Pt_down = np.zeros(pdos_data['Mn1'][low:high,0].shape)
    
    for i in range(4):
        Mn_data = pdos_data['Mn'+str(i)][low:high,0:3]
        plt.plot(Mn_data[:,0],Mn_data[:,1],color = 'black')
        plt.plot(Mn_data[:,0],Mn_data[:,2],color = 'black')
        Mn_up = Mn_up + pdos_data['Mn'+str(i)][low:high,1]
        Mn_down = Mn_down + pdos_data['Mn'+str(i)][low:high,2]
        for j in range(4):
            Pt_data = pdos_data['Pt'+str(j)][low:high,0:3]
            plt.plot(Mn_data[:,0],Pt_data[:,1],color='red')
            plt.plot(Mn_data[:,0],Pt_data[:,2],color = 'red')
            Pt_up = Pt_up + pdos_data['Pt'+str(i)][low:high,1]
            Pt_down = Pt_down + pdos_data['Pt'+str(i)][low:high,2]
            KLD['Mn'+str(i+1)+'-Pt'+str(j+1)+'_up'] = kl_divergence(Mn_data[:,1],Pt_data[:,1])
            KLD['Mn'+str(i+1)+'-Pt'+str(j+1)+'_down'] = kl_divergence(Mn_data[:,2],Pt_data[:,2])
            DIFF['Mn'+str(i+1)+'-Pt'+str(j+1)+'_up'] = simple_difference(Mn_data[:,1],Pt_data[:,1])
            DIFF['Mn'+str(i+1)+'-Pt'+str(j+1)+'_down'] = simple_difference(Mn_data[:,2],Pt_data[:,2])
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('E-E$_F$ (eV)',fontsize =20)
            plt.ylabel('DOS (DOS/eV)',fontsize = 20)
            plt.vlines(0,-4,4,colors='blue',linestyles='--')
            plt.hlines(0,-2,2,colors='blue',linestyles='--')
            plt.savefig(dos_directory+'plots/DOS_all.png')
    
    tot_KLD.append(kl_divergence(Mn_up,Pt_up)+kl_divergence(Mn_down,Pt_down))
    tot_DIFF.append(simple_difference(Mn_up,Pt_up) + simple_difference(Mn_down, Mn_down))
    
fig = plt.figure(figsize=(10,6))
ax = plt.axes((0.2,0.2,0.8,0.8))
plt.scatter(tot_KLD,idmi,s=100)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('KLD (KLD of all interfacial atoms)',fontsize=20)
plt.ylabel('iDMI (meV nm/atom)',fontsize=20)


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