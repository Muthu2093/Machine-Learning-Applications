import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.svm import SVC


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    
    # Include bias in training data
    train_data = np.append(np.ones([n_data,1]),train_data,axis=1)

    theta = sigmoid(np.dot(initialWeights.reshape([1, n_features+1]), np.transpose(train_data)))
    LeftTerm = np.dot(np.log(theta),labeli)
    RightTerm = np.dot(np.log(np.ones(theta.shape) - theta),(np.ones(labeli.shape) - labeli))
    error = (-1/n_data) * (LeftTerm + RightTerm)
    error = sum(error)
    print(error)
    
    error_grad = (1/n_data) * np.reshape((np.dot(np.transpose(train_data), (np.transpose(theta) - labeli))),[716])
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    
    posterior = sigmoid(np.dot(data, W[1:len(W)]))
    for i in range(0,data.shape[0]):
        label[i] = np.where(posterior[i,:] == np.max(posterior[i,:]))
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    
    train_data, labeli = args
    
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    
    initialWeights = params.reshape([n_feature + 1, n_class])
    
    # Include bias in training data
    train_data = np.append(np.ones([n_data,1]),train_data,axis=1)

    theta = np.dot(train_data, initialWeights)
    theta = np.exp(theta)
    theta = theta / np.reshape(sum(np.transpose(theta)),[50000,1]) # check about np.sum
    
    #theta = np.zeros([10,50000])
    #for i in range(0,10):
    #    theta[i,:] = np.reshape(np.exp(np.dot(train_data, np.reshape(initialWeights[:,i],[716,1]))), [50000])
    
    #for i in range(0,10):
    #    theta[i,:] = theta[i,:]/sum(theta)
        
    #theta = np.transpose(theta)

    #error =  np.dot(np.transpose(labeli),np.log(theta))
    #error =  - np.sum(np.diagonal(error))
    #error = error/(labeli.shape[0]*labeli.shape[1])
    
    error = - (np.sum(labeli * theta) * n_data**-1)
    print(error)
    
    error_grad = np.dot(np.transpose(train_data), (theta - labeli)) / (labeli.shape[0]*labeli.shape[1])
    

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad.reshape([(n_feature+1)*n_class])


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    
    posterior = np.exp(np.dot(data, W[1:len(W)]))
    posterior = posterior / sum(posterior) # check about np.sum
    
    for i in range(0,data.shape[0]):
        label[i] = np.where(posterior[i,:] == np.max(posterior[i,:]))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
#W = np.zeros((n_feature + 1, n_class))
#initialWeights = np.zeros((n_feature + 1, 1))
#opts = {'maxiter': 100}
#for i in range(n_class):
#    labeli = Y[:, i].reshape(n_train, 1)
#    args = (train_data, labeli)
#    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#    W[:, i] = nn_params.x.reshape((n_feature + 1,))
#
#print("\n\n--------Binomail LR -----------")
## Find the accuracy on Training Dataset
#predicted_label = blrPredict(W, train_data)
#print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
#
## Find the accuracy on Validation Dataset
#predicted_label = blrPredict(W, validation_data)
#print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
#
## Find the accuracy on Testing Dataset
#predicted_label = blrPredict(W, test_data)
#print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""


##################
# SVM Code Begins here
##################

print('\n\n--------------SVM-------------------\n\n')

def SVM(train_data, train_label, validation_data, validation_label, test_data, test_label, KERNEL, c, flag):
    
    if (flag == False):
        svmModel = SVC(C = c, kernel = KERNEL, gamma = 1.0, verbose = False, cache_size = 1500)
    else:
        svmModel = SVC(C = c, kernel = KERNEL, verbose = False , cache_size = 1500)
    
    #print('\n **** Fitting model on training data ****\n')
    svmModel.fit(train_data, train_label)
    
    #print('\n **** Predicting model on training data ****\n')
    train_predicted = svmModel.predict(train_data)
    
    #print('\n **** Predicting model on validation data ****\n')
    validation_predicted = svmModel.predict(validation_data)
    
    #print('\n **** Predicting model on test data ****\n')
    test_predicted = svmModel.predict(test_data)
    
    Accuracy_train = np.count_nonzero(np.logical_and(train_label, train_predicted.reshape([train_predicted.shape[0],1])))/ train_data.shape[0]
    Accuracy_validation = np.count_nonzero(np.logical_and(validation_label, validation_predicted.reshape([validation_predicted.shape[0],1])))/ validation_data.shape[0]
    Accuracy_test = np.count_nonzero(np.logical_and(test_label, test_predicted.reshape([test_predicted.shape[0],1])))/ test_data.shape[0]
    
    return Accuracy_train,Accuracy_validation, Accuracy_test

## For Linear Kernel
print('\n **** SVM with linear kernel ****\n')
Flag = True
KERNEL = 'linear'
Accuracy_List = []
#[Accuracy_train, Accuracy_validation, Accuracy_test] = SVM(train_data, train_label, validation_data, validation_label, test_data, test_label, KERNEL, 1.0, Flag)
#
#print("Accuracy of train data in SVM: " +str(Accuracy_train))
#print("Accuracy of validation data in SVM: " +str(Accuracy_validation))
#print("Accuracy of test data in SVM: " +str(Accuracy_test))
#Accuracy_List.append(["Linear","Gamme:default","c=1.0","Training_Accuracy:" + Accuracy_train, "Validation_Accuracy:" + Accuracy_validation, "Test_Accuracy:" + Accuracy_test])
# For Radial bias Kernel with Gamma =1
#print('\n **** SVM with linear kernel ****\n')

KERNEL = 'rbf'
GAMMA = 1.0
Flag = False

print('\n **** Radial Bias SVM with gamma = 1 ****\n') ## gamma = 1

#[Accuracy_train, Accuracy_validation, Accuracy_test] = SVM(train_data, train_label, validation_data, validation_label, test_data, test_label, KERNEL, 1.0, Flag)
#Accuracy_List.append(["Radial","Gamme:default","c=1.0","Training_Accuracy:" + Accuracy_train, "Validation_Accuracy:" + Accuracy_validation, "Test_Accuracy:" + Accuracy_test])
#print("Accuracy of train data in SVM: " +str(Accuracy_train))    
#print("Accuracy of validation data in SVM: " +str(Accuracy_validation))    
#print("Accuracy of test data in SVM: " +str(Accuracy_test))


## For Radial bias Kernel with Gamma = default('auto')
print('\n **** Radial Bias SVM with Default Gamma setting ****\n') ## gamma = default
Flag = True
#[Accuracy_train, Accuracy_validation, Accuracy_test] = SVM(train_data, train_label, validation_data, validation_label, test_data, test_label, KERNEL, 1.0, Flag)
#Accuracy_List.append(["Radial","Gamme:1.0","c=1.0","Training_Accuracy:" + Accuracy_train, "Validation_Accuracy:" + Accuracy_validation, "Test_Accuracy:" + Accuracy_test])
#print("Accuracy of validation data in SVM: " +str(Accuracy_validation))    
#print("Accuracy of test data in SVM: " +str(Accuracy_test))


## For Radial bias with varying values of C
print('\n **** Radial Bias SVM with varying C values ****\n') ## gamma = default

C = [50, 60, 70, 80, 90, 100]
for c in C:
    [Accuracy_train, Accuracy_validation, Accuracy_test] = SVM(train_data, train_label, validation_data, validation_label, test_data, test_label, KERNEL, c, Flag)
    #Accuracy_List.append(["Radial","Gamme:default",str(c),"Training_Accuracy:" + Accuracy_train, "Validation_Accuracy:" + Accuracy_validation, "Test_Accuracy:" + Accuracy_test])
    print('C value: ' + str(c))
    print("Accuracy of train data in SVM: " +str(Accuracy_train))    
    print("Accuracy of validation data in SVM: " +str(Accuracy_validation))    
    print("Accuracy of test data in SVM: " +str(Accuracy_test))


file  = open("output.csv",'w+')
for line in Accuracy_List:
    file.write("\n" + str(line))

file.close()

##################
# Multinomail Logistic Regression Code Begins here
##################

print('\n\n--------------Multimomial Logistic Regression-------------------\n\n')

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
