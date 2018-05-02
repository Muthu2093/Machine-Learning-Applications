'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import tensorflow as tf
import numpy as np
import pickle
import timeit
start = timeit.default_timer()

# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron():
    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of features
    #n_hidden_2 = 256  # 2nd layer number of features
    #n_hidden_3 = 256
    #n_hidden_4 = 256
    #n_hidden_5 = 256
    #n_hidden_6 = 256
    #n_hidden_7 = 256
    n_input = 784  # data input
    n_classes = 2

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        #'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        #'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
        #'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
        #'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        #'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        #'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        #'b5': tf.Variable(tf.random_normal([n_hidden_5])),
        #'b6': tf.Variable(tf.random_normal([n_hidden_6])),
        #'b7': tf.Variable(tf.random_normal([n_hidden_7])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    #layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    #layer_4 = tf.nn.relu(layer_4)
    # Hidden layer with RELU activation
    #layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    #layer_5 = tf.nn.relu(layer_5)
    # Hidden layer with RELU activation
    #layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    #layer_6 = tf.nn.relu(layer_6)
    # Hidden layer with RELU activation
    #layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    #layer_7 = tf.nn.relu(layer_7)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer,x,y

# Do not change this
def preprocess():
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    #train_data1 = np.delete(train_data,np.where(np.amax(train_data,axis=0)==0),1)
    #test_data1 = np.delete(test_data,np.where(np.amax(train_data,axis=0)==0),1)
    #validation_data1 = np.delete(validation_data,np.where(np.amax(train_data,axis=0)==0),1)
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


# Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 100

# Construct model
pred,x,y = create_multilayer_perceptron()

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
timer =1
# Initializing the variables
init = tf.global_variables_initializer()

# load data
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = preprocess()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Training cycle
    for epoch in range(training_epochs):
        print(timer)
        timer = timer +1
        avg_cost = 0.
        total_batch = int(train_features.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train_features[i * batch_size: (i + 1) * batch_size], train_labels[i * batch_size: (i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_features, y: test_labels}))
    
    
stop = timeit.default_timer()

print('\n Time Taken: ' + str(stop - start))
