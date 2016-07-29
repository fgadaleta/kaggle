import numpy as np
import re


def load_data():
    # data tranform and loading
    train_data = np.genfromtxt('data/train.csv', dtype=None, delimiter=',', names=True)
    test_data = np.genfromtxt('data/test.csv', dtype=None, delimiter=',', names=True)
    numsamples = len(train_data) 
    numfeatures = 93

    ##### prepare training set #####
    x_train = np.empty([len(train_data), numfeatures])
    y_train = np.empty([len(train_data)])

    # fill input and target with data values
    for i in range(len(train_data)):
        # set the features
        for j in range(numfeatures):
            x_train[i,j] = int(train_data[i][j+1])
    
    for i in range(len(train_data)):
        # set the class label
        label = train_data[i][-1]
        fields = re.split('_', label)
        y_train[i] = int(fields[1])


    # prepare testing set to be submitted
    ##### prepare training set #####
    x_test = np.empty([len(test_data), numfeatures])

    # fill input and target with data values
    for i in range(len(test_data)):
        # set the features
        for j in range(numfeatures):
            x_test[i,j] = int(test_data[i][j+1])

    train_set = (x_train, y_train)
    test_set = (x_test, 0)
    
    rval = [train_set, test_set] 
    return rval
