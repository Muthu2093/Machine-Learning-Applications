import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.svm import SVC
import timeit

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
    
    # Including bias term in training data
    train_data = np.append(np.ones([n_data,1]),train_data,axis=1)

    # Estimating posterior
    theta = sigmoid(np.dot(initialWeights.reshape([1, n_features+1]), np.transpose(train_data)))
    LeftTerm = np.dot(np.log(theta),labeli)
    RightTerm = np.dot(np.log(np.ones(theta.shape) - theta),(np.ones(labeli.shape) - labeli))
    
    # Estimating error
    error = (-1/n_data) * (LeftTerm + RightTerm)
    error = sum(error)
    print(error) # uncomment line to visualize gradient descent
    
    # Estimating gradients
    error_grad = (1/n_data) * np.reshape((np.dot(np.transpose(train_data), (np.transpose(theta) - labeli))),n_features + 1)

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
    ## Including bias term to input data
    data = np.append(np.ones([data.shape[0],1]),data,axis=1)
    label = np.zeros((data.shape[0], 1))
    
    ## Estimating Posterior
    posterior = np.exp(np.dot(data, W))
    posterior = posterior / np.reshape(sum(np.transpose(posterior)),[data.shape[0],1]) # check about np.sum
    
    ## Extracting label of Maximum Posterior
    for i in range(0,data.shape[0]):
        label[i] = np.where(posterior[i,:] == np.max(posterior[i,:]))

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
    
    # Including bias in training data
    train_data = np.append(np.ones([n_data,1]),train_data,axis=1)

    # Estimating posterior 
    theta = np.dot(train_data, initialWeights)
    theta = np.exp(theta)
    theta = theta / np.reshape(sum(np.transpose(theta)),[n_data,1]) # check about np.sum
    
    # Estimating error
    error =  np.dot(np.transpose(labeli),np.log(theta))
    error =  - np.sum(np.diagonal(error))
    error = error/(labeli.shape[0]*labeli.shape[1])
    print(error) # - uncomment to visualize the gradient descent
    
    # Estimating gradients
    error_grad = np.dot(np.transpose(train_data), (theta - labeli)) / (labeli.shape[0]*labeli.shape[1])
    error_grad = error_grad.reshape([(n_feature+1)*n_class])
    
    return error, error_grad


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
    ## Adding bias term to input data
    data = np.append(np.ones([data.shape[0],1]),data,axis=1)
    label = np.zeros((data.shape[0], 1))
    
    ## Calculation Posterior
    posterior = np.exp(np.dot(data, W))
    posterior = posterior / np.reshape(sum(np.transpose(posterior)),[data.shape[0],1]) # check about np.sum
    
    ## Extracting label of Maximum Posterior
    for i in range(0,data.shape[0]):
        label[i] = np.where(posterior[i,:] == np.max(posterior[i,:]))

    return label

def confusionMatrix(label, predict):
    CF = np.zeros([10,10])
    
    for i in range(0,len(label)):
        CF[int(label[i]),int(predict[i])] += 1
        
    classAccuracy = np.zeros(10)
    for i in range (0,10):
        classAccuracy[i]= CF[i,i] * 100/ np.sum(CF[i,:])
    print("Confusion Matrix:")
    CF = CF.astype(int)
    df_cm = pd.DataFrame(CF.astype(int), index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    
    return CF, classAccuracy


"""
Script for Logistic Regression
"""

start = timeit.default_timer()

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
    
Accuracy_List = []

"""
Script for Binomial Logistic Regression
"""
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

print("\n\n--------Binomail LR -----------")
# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
[confMat_BLR, ClassAccBLR] = confusionMatrix(test_label, predicted_label)

stop = timeit.default_timer()
time = stop - start
print ("Time Taken = " + str(time) +" seconds")

"""
Script for Support Vector Machine
"""


##################
# SVM Code Begins here
##################

print('\n\n--------------SVM-------------------\n\n')
## For Linear Kernel
print('\n **** SVM with linear kernel ****\n')

start = timeit.default_timer()
svmModel = SVC(kernel = "linear")
svmModel.fit(train_data, train_label)
#train_predicted = svmModel.predict(train_data)
#validation_predicted = svmModel.predict(validation_data)
#test_predicted = svmModel.predict(test_data)
    
Accuracy_train = svmModel.score(train_data,train_label)
Accuracy_validation = svmModel.score(validation_data,validation_label)
Accuracy_test = svmModel.score(test_data, test_label)
  
print("Accuracy of train data in SVM: " +str(Accuracy_train))
print("Accuracy of validation data in SVM: " +str(Accuracy_validation))
print("Accuracy of test data in SVM: " +str(Accuracy_test))
Accuracy_List.append(["Linear","Gamme = default","c = 1.0","Training_Accuracy:" + str(Accuracy_train), "Validation_Accuracy:" + str(Accuracy_validation), "Test_Accuracy:" + str(Accuracy_test)])

stop = timeit.default_timer()
time = stop - start
print ("Time Taken = " + str(time) +" seconds")



#For Radial bias Kernel with Gamma =1
start = timeit.default_timer()

print('\n **** Radial Bias SVM with gamma = 1 ****\n') ## gamma = 1
svmModel = SVC(kernel = "rbf", gamma = 1)
svmModel.fit(train_data, train_label)
#train_predicted = svmModel.predict(train_data)
#validation_predicted = svmModel.predict(validation_data)
#test_predicted = svmModel.predict(test_data)
    
Accuracy_train = svmModel.score(train_data,train_label)
Accuracy_validation = svmModel.score(validation_data,validation_label)
Accuracy_test = svmModel.score(test_data, test_label)
  
Accuracy_List.append(["Radial","Gamme = default","c = 1.0","Training_Accuracy:" + str(Accuracy_train), "Validation_Accuracy:" + str(Accuracy_validation), "Test_Accuracy:" + str(Accuracy_test)])
print("Accuracy of train data in SVM: " +str(Accuracy_train))    
print("Accuracy of validation data in SVM: " +str(Accuracy_validation))    
print("Accuracy of test data in SVM: " +str(Accuracy_test))

stop = timeit.default_timer()
time = stop - start




### For Radial bias Kernel with Gamma = default('auto')
print('\n **** Radial Bias SVM with Default Gamma setting ****\n') ## gamma = default
start = timeit.default_timer()
svmModel = SVC(kernel = "rbf")
svmModel.fit(train_data, train_label)
#train_predicted = svmModel.predict(train_data)
#validation_predicted = svmModel.predict(validation_data)
#test_predicted = svmModel.predict(test_data)
    
Accuracy_train = svmModel.score(train_data,train_label)
Accuracy_validation = svmModel.score(validation_data,validation_label)
Accuracy_test = svmModel.score(test_data, test_label) 

Accuracy_List.append(["Radial","Gamme = 1.0","c = 1.0","Training_Accuracy:" + str(Accuracy_train), "Validation_Accuracy:" + str(Accuracy_validation), "Test_Accuracy:" + str(Accuracy_test)])
print("Accuracy of train data in SVM: " +str(Accuracy_train))    
print("Accuracy of validation data in SVM: " +str(Accuracy_validation))    
print("Accuracy of test data in SVM: " +str(Accuracy_test))


stop = timeit.default_timer()
time = stop - start
print ("Time Taken = " + str(time) +" seconds")




## For Radial bias with varying values of C
print('\n **** Radial Bias SVM with varying C values ****\n') ## gamma = default
Flag = True;
C = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
for c in C:
    start = timeit.default_timer()
    svmModel = SVC(C = c, kernel = "rbf")
    svmModel.fit(train_data, train_label)
    #train_predicted = svmModel.predict(train_data)
    #validation_predicted = svmModel.predict(validation_data)
    #test_predicted = svmModel.predict(test_data)
    
    Accuracy_train = svmModel.score(train_data,train_label)
    Accuracy_validation = svmModel.score(validation_data,validation_label)
    Accuracy_test = svmModel.score(test_data, test_label)
  
    Accuracy_List.append(["Radial","Gamme:default","c = " + str(c),"Training_Accuracy:" + str(Accuracy_train), "Validation_Accuracy:" + str(Accuracy_validation), "Test_Accuracy:" + str(Accuracy_test)])
    print('C value: ' + str(c))
    print("Accuracy of train data in SVM: " +str(Accuracy_train))    
    print("Accuracy of validation data in SVM: " +str(Accuracy_validation))    
    print("Accuracy of test data in SVM: " +str(Accuracy_test))
    stop = timeit.default_timer()
    time = stop - start
    print ("Time Taken = " + str(time) +" seconds")

file  = open("output.csv",'w+')
for line in Accuracy_List:
    file.write("\n" + str(line))

file.close()



##################
# Multinomail Logistic Regression Code Begins here
##################

print('\n\n--------------Multimomial Logistic Regression-------------------\n\n')

start = timeit.default_timer()
"""
Script for Extra Credit Part
"""
## FOR EXTRA CREDIT ONLY
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
stop = timeit.default_timer()
time = stop - start
print ("Time Taken = " + str(time) +" seconds")

[confMat_MLR, ClassAccMLR] = confusionMatrix(test_label, predicted_label_b)