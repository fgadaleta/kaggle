from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import cPickle
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



def save(obj, fname, path='./tmp/', verbose=False):
    """ Save whatever passed as argument to file """
    assert isinstance(fname, str)
    
    # create path if not exist
    try:
        os.stat(path)
    except:
        os.mkdir(path) 
        
    fname = path+fname

    if verbose:
        print("Saving to file [%s]"%fname)

    f = file(fname, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return 0


def load(objname, path='./tmp/'):
    """ Load objname from path and returns data structure """
    assert isinstance(objname, str)
    assert os.stat(path)
    
    fname = path+objname
    f = file(fname, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj
    


def writePredictionFile(predictions, modeltype="default"):
    # prepare prediction for submission 
    filename = "results/submit-%s.csv"%(modeltype)
    f = open(filename,'w')
    f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")

    for i in range(len(predictions)):
        label = predictions[i]
        line = str(i+1)
        for j in range(1,10):
            if j == int(label):
                line = line + ",1"
            else:
                line = line + ",0"
            
        f.write(line+"\n") # python will convert \n to os.linesep
    f.close()        # you can omit in most cases as the destructor will call if



def load_data(path, train=True, selected=False):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        #X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]

        if isinstance(selected, np.ndarray):
            X, labels = X[:, 1+selected].astype(np.float32), X[:, -1]
        else:
            X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        
        return X, labels
    else:
        #X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)

        if isinstance(selected, np.ndarray):
            X, ids = X[:, 1+selected].astype(np.float32).astype(np.float32), X[:, 0].astype(str)
        else:
            X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)   
            
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

# FG
def add_features(X):
    nfeats = X.shape[1]
    nextra = nfeats*(nfeats-1)/2
    ntotal = nfeats + nextra
    Xplus = np.zeros([X.shape[0],ntotal])
    
    for row in range(X.shape[0]):
        i = 0
        extra = np.zeros(nextra)
        # pairwise n(n-1)/2 products 
        for col in range(X.shape[1]):
            for succ in range(col+1, nfeats):
                extra[i] = X[row, col]*X[row,succ]
                i += 1

        Xplus[row,0:nfeats] = X[row,]
        Xplus[row,nfeats:] = extra

    return Xplus    
    
def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        #encoder.fit(labels)
        encoder.fit(y)
    #y = encoder.transform(labels).astype(np.int32)
    y = encoder.transform(y).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

def ensemble(prediction_files, weights=None):
    """
    Given a list of prediction_files computes the ensemble 
    each model is weighted with weights if any
    """

    preds = []
    
    for filename in prediction_files:
        with open(filename, 'rb') as csvfile:
            filereader = csv.DictReader(csvfile)
            #rows = sum(1 for r in filereader)
            #data = np.zeros(shape=(rows, 9))
            data = np.zeros(shape=(144368, 9))
            
            for idx,row in enumerate(filereader):
                #for row in filereader:
                cnvinfo = [row['Class_1'], row['Class_2'], row['Class_3'],
                           row['Class_4'], row['Class_5'],row['Class_6'], 
                           row['Class_7'], row['Class_8'], row['Class_9'] ]
                data[idx,] = cnvinfo

        # add this to prediction list
        preds.append(data)

    # weight and merge preds 
    res = np.zeros(shape=(144368, 9))
    
    for sample in range(144368):
        for i in range(len(preds)):
            res[sample] += weights[i]*preds[i][sample] #+ weights[1]*preds[1][sample]
    
    return res
