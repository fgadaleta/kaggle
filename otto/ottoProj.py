import cPickle
import gzip
import os
import sys
import time
import re
import numpy as np

import theano
import theano.tensor as T


# data tranform and loading
train_data = np.genfromtxt('data/train.csv', dtype=None, delimiter=',', names=True)
numsamples = len(train_data) 
numfeatures = 93

# split train set in train+valid
data_idx = np.arange(numsamples)

# shuffle the indices not the data!
np.random.shuffle(data_idx)    
train_idx, test_idx, valid_idx = np.array_split(data_idx,3)
train_sample = train_data[train_idx]
test_sample  = train_data[test_idx]
valid_sample = train_data[valid_idx]

#train_set, valid_set, test_set format: tuple(input, target)
#input is an numpy.ndarray of 2 dimensions (a matrix)
#witch row's correspond to an example. target is a
#numpy.ndarray of 1 dimensions (vector)) that have the same length as
#the number of rows in the input. It should give the target
#target to the example with the same index in the input.
    

# prepare training set
input = np.empty([len(train_sample), numfeatures])
target = np.empty([len(train_sample)])

# fill input and target with data values
for i in range(len(train_sample)):
    # set the features
    for j in range(numfeatures):
        input[i,j] = int(train_sample[i][j+1])
    
for i in range(len(train_sample)):
    # set the class label
    label = train_sample[i][-1]
    fields = re.split('_', label)
    target[i] = int(fields[1])
    
# pack everything in a tuple
train_set = (input,target)



# prepare test set
input = np.empty([len(test_sample), numfeatures])
target = np.empty([len(test_sample)])

# fill input and target with data values
for i in range(len(test_sample)):
    # set the features
    for j in range(numfeatures):
        input[i,j] = int(test_sample[i][j+1])
    
for i in range(len(test_sample)):
    # set the class label
    label = test_sample[i][-1]
    fields = re.split('_', label)
    target[i] = int(fields[1])
    
# pack everything in a tuple
test_set = (input,target)



# prepare valid set
input = np.empty([len(valid_sample), numfeatures])
target = np.empty([len(valid_sample)])

# fill input and target with data values
for i in range(len(valid_sample)):
    # set the features
    for j in range(numfeatures):
        input[i,j] = int(valid_sample[i][j+1])
    
for i in range(len(valid_sample)):
    # set the class label
    label = valid_sample[i][-1]
    fields = re.split('_', label)
    target[i] = int(fields[1])
    
# pack everything in a tuple
valid_set = (input,target)







 
