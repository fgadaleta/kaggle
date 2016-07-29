from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import utils as ut
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.

    Compatible Python 2.7-3.4 

    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.

    Best validation score at epoch 21: 0.4881 

    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
'''

np.random.seed(1337) # for reproducibility

sort_idx = ut.load("sort_idx")
sort_idx = sort_idx[-93:]
print(type(sort_idx))
print(sort_idx.shape)


## check if raw data exist
if os.path.isfile("./tmp/X-extra-features-0"):
    print ("Loading existing data (with extra features)...")
    #print(X.shape)

    for i in range(6):
        print("Loading subset %d"%i)
        objname = "X-extra-features-%d"%i
        labelsname = "labels-%d"%i
        if i==0:
            X = ut.load(objname)
            labels = ut.load(labelsname)
        else:
            X = np.concatenate((X, ut.load(objname)))
            labels = np.concatenate((labels, ut.load(labelsname)))
    
else:
    print("Loading data...")
    X, labels = ut.load_data('data/train.csv', train=True, selected=sort_idx)
    
    
dims = X.shape
print(dims, 'dims')
    
########################################################################
print("Preprocessing data")
X, scaler = ut.preprocess_data(X)
print("Preprocessing labels")
y, encoder = ut.preprocess_labels(labels)

X_test, ids = ut.load_data('data/test.csv', train=False, selected=sort_idx)
X_test, _ = ut.preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

## check if model exists and resume otherwise rebuild
if os.path.isfile("./tmp/keras-nn"):
    print ("Loading existing neural network...")
    model = ut.load("keras-nn", "./tmp/")
    print ("done.")
else:
    print("Building model...")

    model = Sequential()
    model.add(Dense(dims, 612))
    model.add(PReLU((612,)))
    model.add(BatchNormalization((612,)))
    model.add(Dropout(0.5))
    
    model.add(Dense(612, 612, init='glorot_uniform'))
    model.add(PReLU((612,)))
    model.add(BatchNormalization((612,)))
    model.add(Dropout(0.5))
    
    model.add(Dense(612, 612, init='glorot_uniform'))
    model.add(PReLU((612,)))
    model.add(BatchNormalization((612,)))
    model.add(Dropout(0.5))
    
    model.add(Dense(612, nb_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    #model.compile(loss='categorical_crossentropy', optimizer="sgd")
    

print("Training model...")
ne = 17
bs = 32
vs = 0.15
model.fit(X, y, nb_epoch=ne, batch_size=bs, validation_split= vs)

print ("Saving model (will overwrite existing one)")
filename = "keras-nn-%d-%d-%d"%(ne,bs,vs)
ut.save(model, filename, verbose=True)

print("Generating submission...")
proba = model.predict_proba(X_test)
ut.make_submission(proba, ids, encoder, fname='keras-otto-proba-93.csv')

#print(type(proba))
#print(proba[0:10,])

