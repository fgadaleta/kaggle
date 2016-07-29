import numpy as np 
import re
import matplotlib.pyplot as plt


def load_data(datapath, datatype="train", split=[0.70,0.20,0.10], verbose=False):
    ''' Loads the dataset
    if training data 
    split the data into train/test/valid
    :type dataset: string
    Return list [train_set, test_set, valid_set]
    (train_set,valid_set,test_set are a split of original train set)
    
    if testing data
    Return matrix
    '''


    def addExtraFeat(x):
        print "Adding extra features"
        # add extra features
        numsamples, numfeatures= x.shape
        extra = np.zeros(numfeatures-7)
        x_ext = np.empty([numsamples, numfeatures+len(extra)])

        for row in range(numsamples):
            extra = np.empty(numfeatures-7)
            for i in range(numfeatures-7):
                extra[i] =  np.sum(x[row, i:i+7]) 
            
            x_ext[row, 0:93] = x[row,]
            x_ext[row, numfeatures:] = extra
    
        return x_ext

    def transformFeat(x, power=0.3):
        print "Transforming to power %f"%power
        return x**power


    def cartesianProduct(x):
        print "Computing cartesian product"
        numfeats = numfeatures + numfeatures**2
        x_ext = np.empty([numsamples, numfeats])
        
        #normalise 
        #row_sums = x.sum(axis=1)
        #x_sc = x/row_sums[:, np.newaxis]
        
        #cartesian product of row and add extra features
        for row in xrange(numsamples):
            if (row%5000 == 0):
                print row

            extra = np.empty(numfeatures**2)
            cart = [(x0,y0) for x0 in x[row,] for y0 in x[row,] ]
            for i in xrange(len(cart)):
                extra[i] = cart[i][0]*cart[i][1]
            
            x_ext[row, 0:93] = x[row,]
            x_ext[row, numfeatures:] = extra
            
        return x_ext
        
        
    ##################
    # LOAD OTTO DATA #
    ##################
    
    if(datatype=="train"):
        
        if verbose:
            print "Loading training data set..."
        
        # data tranform and loading
        train_data = np.genfromtxt(datapath, dtype=None, delimiter=',', names=True)
        # load test_data if datatype='test' do not add the class
        #test_data = np.genfromtxt('data/test.csv', dtype=None, delimiter=',', names=True)
        numsamples = len(train_data) 
        # split train set in train+valid
        data_idx = np.arange(numsamples)
        
        if verbose:
            print "Splitting training set [train %f, test %f, valid %f]"%(split[0], split[1], split[2])
        
        # shuffle the indices not the data!
        np.random.shuffle(data_idx)    
        trainlen = np.ceil(split[0]*numsamples)
        testlen  = np.ceil(split[1]*numsamples)
        validlen = np.ceil(split[2]*numsamples)
        train_idx = data_idx[:trainlen]
        test_idx  = data_idx[trainlen:trainlen+testlen]
        valid_idx = data_idx[trainlen+testlen:]
        
        # equally splits data
        #train_idx, test_idx, valid_idx = np.array_split(data_idx,3)
    
        train_sample = train_data[train_idx]
        test_sample  = train_data[test_idx]
        valid_sample = train_data[valid_idx]
        
        ##### prepare training set #####
        if verbose:
            print "Preparing training set"
            
        numfeatures = 93  # TODO read this from data shape
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
        #feats = transformFeat(addExtraFeat(input))
        #feats = cartesianProduct(input)
        train_set = (input,target)
        
        ##### prepare test set #####
        if verbose:
            print "Preparing testing set"
        
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
        #feats = addExtraFeat(input)
        #feats = cartesianProduct(input)
        test_set = (input,target)
        
        
        ##### prepare valid set #####
        if verbose:
            print "Preparing validation set"
        
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
        #feats = addExtraFeat(input)
        #feats = cartesianProduct(input)
        valid_set = (input,target)
        rval = [train_set, test_set, valid_set]
        
    elif(datatype=="test"):
        if verbose:
            print "Loading new data set (unseen/testing)..."
        
        numfeatures = 93
        unseen_data = np.genfromtxt(datapath, dtype=None, delimiter=',', names=True)
        
        input = np.empty([len(unseen_data), numfeatures])
        
        for i in range(len(unseen_data)):
            for j in range(numfeatures):
                input[i,j] = int(unseen_data[i][j+1])
        
        rval = (input)
            
    return rval
