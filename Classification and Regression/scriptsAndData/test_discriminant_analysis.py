#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:23:18 2018

@author: muthuvel
"""
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

c = np.hstack((Xtest,ytest))  
classes, count = np.unique(ytest[:,0], return_counts = True)
theta = np.zeros ((len(classes)))
label = np.zeros(np.shape(ytest))
eff = np.zeros(np.shape(ytest))
D = np.zeros((len(classes)))

## Calculating theta(y=1) for test data
for i in range (0,1):#len(classes)):
    theta[i] = count[i]/len(ytest)
    
for i in range (0,len(ytest)):    
    X = np.matlib.repmat(Xtest[i,:],len(classes),1)    
    nu = np.zeros(np.shape(X))
    #m = mean[:,i]
    #nu = np.matlib.repmat(m,len(X),1)
    nu = np.transpose(mean)
    sigma = inv(coVariance)
    D =  np.matmul(np.matmul((X - nu),sigma),np.transpose(X-nu)) #covmat - change variable name
    pdf = np.amin(np.diagonal(D))
    l= np.where(np.diagonal(D) == pdf)
    label[i]=classes[l]
    if label[i] == ytest[i]:
        eff[i] = 1
    else:
        eff[i] = 0
    print(i)
    
unique, counts = np.unique(eff, return_counts=True)
eff=counts[1]/len(eff)*100


