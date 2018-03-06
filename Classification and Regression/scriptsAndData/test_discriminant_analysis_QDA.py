#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:04:34 2018

@author: muthuvel
"""

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
eff = np.zeros([len(ytest),len(classes)])
acc = np.zeros(len(ytest))
D = np.zeros((len(classes)))

    ## Calculating theta(y=1) for test data - all classes
for i in range (0,(len(classes))):#len(classes)):
    theta[i] = count[i]/len(ytest)
        
for i in range (0,len(classes)):    
    #X = np.matlib.repmat(Xtest[i,:],len(classes),1)    
    #nu = np.zeros(np.shape(X))
    #m = mean[:,i]
    #nu = np.matlib.repmat(m,len(X),1)
    #nu = np.transpose(means)
    #sigma = inv(covmats)
    #D =  np.matmul(np.matmul((X - nu),sigma),np.transpose(X-nu)) #covmat - change variable name
    #pdf = np.amin(np.diagonal(D))
    #l= np.where(np.diagonal(D) == pdf)
    #label[i]=classes[l]
    #if label[i] == ytest[i]:
    #    acc[i] = 1
    #else:
    #    acc[i] = 0
    #print(i)
    X=Xtest
    nu=np.matlib.repmat(mean[:,i],len(Xtest),1)
    sigma=inv(coVariance[i])   ## change coVariance to covmats
    D=np.matmul(np.matmul((X-nu),sigma),np.transpose(X-nu))
    eff[:,i] = np.diagonal(D)

for i in range (0,len(ytest)):    
    l=np.where(eff[i,:]==np.amin(eff[i,:]))
    label[i]=classes[l]
    if label[i]==ytest[i]:
        acc[i]=1
    else:
        acc[i]=0
        
unique, counts = np.unique(acc, return_counts=True)
accuracy=counts[1]/len(acc)*100


