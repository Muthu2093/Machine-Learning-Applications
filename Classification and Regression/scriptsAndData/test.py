#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:33:22 2018

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


if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

c=np.hstack((X,y))    
UE  = np.unique (y[:,0])
mean = np.zeros ((np.shape(X)[1], len(UE)))
coVariance = [0]* len(UE)
temp = np.zeros ((np.shape(X)[1], np.shape(X)[1]))
    
    ## Calculation of mean
for i in range (0,len(UE)):
    for j in range (0,np.shape(X)[1]):
        elemIndices = np.where (c[:,2] == UE[i])
        classElem = X[elemIndices,j]
        mu = sum (classElem[0,:])
        mu = mu/np.shape (classElem)[1]
        mean[j,i] = mu
        
    ## Calculation of covariance matrix
    nu = np.matlib.repmat(mu,len(X),1)
    D = np.matmul(np.transpose(X-nu),(X - nu))
    D = D/len(X)
    coVariance[i]=D
    
#del i,j,elemIndices,mu,squareDiff,classElem, pi,X,y,c


        
        
        


