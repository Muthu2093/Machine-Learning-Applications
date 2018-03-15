import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    c=np.hstack((X,y))    
    UE  = np.unique (y[:,0])
    mean = np.zeros ((np.shape(X)[1], len(UE))) #initializing mean matrix
    coVariance = np.zeros ((np.shape(X)[1], np.shape(X)[1])) #initialzing covariance matrix
    
    ## Calculation of mean
    for i in range (0,len(UE)):
        for j in range (0,np.shape(X)[1]):
            elemIndices = np.where (c[:,2] == UE[i])
            classElem = X[elemIndices,j]
            mean[j,i]= np.mean(classElem[0,:]) 
    
    ##Caculation covariance 
    coVariance = np.cov(np.transpose(X))
    
    return mean,coVariance

def qdaLearn(X,y):
    
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
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
        
        ## Calulating covariance
        nu = np.matlib.repmat(mu,len(X),1)
        D = np.matmul(np.transpose(X-nu),(X - nu))
        D = D/len(X)
        coVariance[i]=D
            
    return mean,coVariance

def ldaTest(means,covmat,Xtest,ytest):
 
    classes, count = np.unique(ytest[:,0], return_counts = True)
    theta = np.zeros (np.shape(means)[1])
    label = np.zeros([len(Xtest),1])
    eff = np.zeros([len(Xtest),np.shape(means)[1]])
    acc = np.zeros(len(Xtest))
    D = np.zeros(np.shape(means)[1])
    a = np.array([1,2,3,4,5])

    for i in range (0,np.shape(means)[1]):    
        X=Xtest
        nu=np.matlib.repmat(means[:,i],len(Xtest),1)
        sigma=inv(covmat)
        D=np.matmul(np.matmul((X-nu),sigma),np.transpose(X-nu)) #pdf for all samples for a specific class
        eff[:,i] = np.diagonal(D)
        
    for i in range (0,len(Xtest)):    
        l=np.where(eff[i,:]==np.amin(eff[i,:]))
        label[i]=a[l]
        if label[i]==ytest[i]:
            acc[i]=1
        else:
            acc[i]=0
    
    # Calculating accuracy    
    unique, counts = np.unique(acc, return_counts=True)
    accuracy=np.count_nonzero(acc==1)/len(acc)*100 
    
    return accuracy,label

def qdaTest(means,covmats,Xtest,ytest):
    
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    
    classes, count = np.unique(ytest[:,0], return_counts = True)
    theta = np.zeros (np.shape(means)[1])
    label = np.zeros([len(Xtest),1])
    eff = np.zeros([len(Xtest),np.shape(means)[1]])
    acc = np.zeros(len(Xtest))
    D = np.zeros(np.shape(means)[1])
    a = np.array([1,2,3,4,5])

    for i in range (0,np.shape(means)[1]):
        X=Xtest
        nu=np.matlib.repmat(means[:,i],len(Xtest),1)
        sigma=inv(covmats[i])
        D=np.matmul(np.matmul((X-nu),sigma),np.transpose(X-nu)) #pdf for all samples for a specific class
        eff[:,i] = np.diagonal(D)
        
    for i in range (0,len(Xtest)):    
        l=np.where(eff[i,:]==np.amin(eff[i,:]))
        label[i]=a[l]
        if label[i]==ytest[i]:
            acc[i]=1
        else:
            acc[i]=0
    
    # Calculating accuracy
    unique, counts = np.unique(acc, return_counts=True)
    accuracy=np.count_nonzero(acc==1)/len(acc)*100
    
    return accuracy,label

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                 

     w = np.matmul(inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),y))
     
     return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1  
    
    I= np.eye((np.shape(X)[1]))
    lambdI= I * lambd
    Inv=inv(lambdI+np.matmul(np.transpose(X),X))
    w = np.matmul(Inv,np.matmul(np.transpose(X),y))
                                                 
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    mse =0
    for i in range (0,np.shape(ytest[:,0])[0]):
        #w = np.squeeze((w))
        #temp = Xtest[i]
        value = ytest[i] - np.dot(np.squeeze(np.asarray(w)),Xtest[i])
        value = np.dot(value,value)
        
        if i == 0:
            mse = value;
        if i != 0:
            mse = mse +value
            
    mse = mse/len(ytest)
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda 
    
    if (np.shape(w)[0]==np.shape(Xtest)[1]):
        w=w
    else:
        w=np.transpose(w)
    
    A = np.subtract(y,np.reshape(np.dot(X,w),[len(y),1]))
    B = lambd * np.dot(np.transpose(w),w)
    Jw =np.dot(np.transpose(A),A)/2 + B/2 # Do not divide by 2 for smooth curve
    error=Jw
    
    error_grad= np.dot(-2 * np.transpose(X),A) + np.reshape(np.dot(2,np.dot(lambd, w)),[np.shape(X)[1],1])
    error_grad=np.squeeze(np.asarray(error_grad))   
                               
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 

    x=np.reshape(x,[len(x),1])
    intercept=np.ones([len(x),1])
    Xp= intercept
    for i in range(1,p+1):
        temp = np.reshape(np.power(x,i),[len(x),1])
        Xp = np.concatenate((Xp,temp),axis=1)
    
    return Xp

# Main script

# Problem 1
print("######## PROBLEM 1 #########")
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('QDA')

plt.show()


# Problem 2
print("######## PROBLEM 2 #########")
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
print("######## PROBLEM 3 #########")
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if i==6:
        weight_ideal=w_l
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

# Problem 4
print("######## PROBLEM 4 #########")
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
print("######## PROBLEM 5 #########")
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)] #  lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
