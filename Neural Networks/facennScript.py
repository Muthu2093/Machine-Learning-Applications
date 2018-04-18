'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle

import timeit
start = timeit.default_timer()

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    Z = np.array(z)
    if (np.shape(Z) == ()):
        sigmoid = 1/(1+math.exp(-Z))
        return sigmoid
    else:
        sigmoid = 1/(1+np.exp(-Z))
        return sigmoid
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""


    # Your code here
    #
    #
    #
    #
    #



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    ## Creating bias  term in training data
    training_data = np.concatenate((training_data,np.reshape(np.ones(len(training_data)),[len(training_data), 1])),axis = 1)
    
    ## Feed Forward network
    output_HL = np.dot(w1,np.transpose(training_data))
    output_HL_Sigmoid = sigmoid(output_HL)
    bias_hidden = np.ones([1,output_HL_Sigmoid.shape[1]])
    output_HL_Sigmoid = np.concatenate((output_HL_Sigmoid,bias_hidden),axis = 0)
    output_OL = np.dot(w2,output_HL_Sigmoid)
    output_OL_Sigmoid = sigmoid(output_OL)
    
    ## Creating boolean labels with size k * n
    training_label_n = np.zeros([n_class, training_label.shape[0]])
    for i in range(0,len(training_label)):
        training_label_n[int(training_label[i]), i ] = 1
    
    ## Calculating error_function
    leftTerm =   training_label_n * np.log(output_OL_Sigmoid)  
    rightTerm = (1-training_label_n) * np.log(1-output_OL_Sigmoid)
    error_OL = leftTerm + rightTerm
    error_OL = -np.sum(error_OL)/training_data.shape[0]
    
    ## Calculating gradients for w1 and w2
    delta_OL = output_OL_Sigmoid - training_label_n
    grad_w2 = np.matmul(delta_OL, np.transpose(output_HL_Sigmoid))
    
    leftTerm = np.dot(np.transpose(w2), delta_OL)
    rightTerm = output_HL_Sigmoid*(1 - output_HL_Sigmoid)
    grad_w1 =  np.dot(leftTerm*rightTerm, training_data)
    
    ## Incorportaing regularization in gradients and and in error function
    grad_w1 = (grad_w1[0:n_hidden,:] + (lambdaval * w1))/ len(training_data)
    grad_w2 = (grad_w2 + (lambdaval * w2))/ len(training_data)
    
    obj_val = error_OL + (lambdaval/(2*len(training_data))) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    
    print(obj_val)
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val, obj_grad)


    
# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    data = np.concatenate((data,np.reshape(np.ones(len(data)),[len(data), 1])),axis = 1)
    
    output_HL = np.dot(w1,np.transpose(data))
    output_HL_Sigmoid = sigmoid(output_HL)
    bias_hidden = np.ones([1,output_HL_Sigmoid.shape[1]])
    output_HL_Sigmoid = np.concatenate((output_HL_Sigmoid,bias_hidden),axis = 0)
    output_OL = np.dot(w2,output_HL_Sigmoid)
    output_OL_Sigmoid = sigmoid(output_OL)
    
    labels = np.array([])
    labels = np.argmax(output_OL_Sigmoid,axis=0)
    # Your code here

    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

stop = timeit.default_timer()

print('\n Time Taken:'+str(stop - start))