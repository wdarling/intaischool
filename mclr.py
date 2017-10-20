# mclr.py
#
# Multi-class Logistic Regression

import os
import sys
import numpy as np

num_classes = 10

data_dir = os.path.join("data", "CIFAR")
train_file = os.path.join(data_dir, "Train_cntk_text.txt")
test_file = os.path.join(data_dir, "Test_cntk_text.txt")

# set up class map
class_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load(filename):
    X = []
    Y = []
    
    for line in open(filename):
        data = line.split()
        
        Y.append(data[1:11])
        X.append(data[12:])
    
    # add 1 for bias...
    X = np.hstack([np.matrix(np.ones(len(X))).T, X])
    
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32).reshape(len(Y), num_classes)
    
def softmax(x):
    
    if x.ndim == 1:
        e = np.exp(x - np.max(x))  # to avoid overflow
        return e / np.sum(e, axis=0)
    else:
        e = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
        return e / np.array([np.sum(e, axis=1)]).T

def sgd(X, Y, epochs=10, lr=0.1):
    losses = []
    
    W = np.random.rand(X.shape[1], num_classes)
   
    for i in range(epochs):
        for j in range(Y.shape[0]):
            # get a single example
            x = X[j,:]
            error = softmax(np.dot(x, W)) - Y[j,:]
            delta = np.outer(x, error)
            W = W - delta * lr
            
        # compute the new loss
        full_error = softmax(np.dot(X, W)) - Y
        loss = np.sum(np.square(full_error)) / Y.shape[0]
        print(loss)
        losses.append(loss)
        
    return W, losses

def main():

    # load the training data and normalize it
    # BIG NOTE! I almost guarantee that loading the training file will not work like this because it requires so much memory.
    # One idea is to look into np.memmap. Another idea is to dynamically load up pieces of the file within the training loop.
    # Finally, why not use CNTK's data readers? They supply you with minibatch data in numpy form and you get all kinds of 
    # other goodies for free!
    print('Loading training data...')
    features_train, labels_train = load(train_file)
    features_train = features_train / 255.0

    # run SGD to train the model
    print('Training model...')
    W, L = sgd(features_train, labels_train, epochs=50)

    # now test it
    features_test, labels_test = load(test_file)
    features_test = features_test / 255.0

    # test...

if __name__ == '__main__':
    main()