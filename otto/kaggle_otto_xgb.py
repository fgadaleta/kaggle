from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import utils as ut
import os
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(1337) # for reproducibility

## check if raw data exist
print("Loading data...")
X, labels = ut.load_data('data/train.csv', train=True)
data, ids = ut.load_data('data/test.csv', train=False)

print("Preprocessing data")
X, scaler = ut.preprocess_data(X)
data, datascaler = ut.preprocess_data(data)

print("Preprocessing labels")
y, encoder = ut.preprocess_labels(labels)

print(y[0:10])
labels = []
for i in range(len(y)):
    labels.append(np.where(y[i]==1)[0][0])
print(labels[0:10])

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape
print(dims, 'dims')

dtrain = xgb.DMatrix(X[0:40000,], label=labels[0:40000])
dtest = xgb.DMatrix(X[40000:, ], label=labels[40000:])
    
#param = {'bst:max_depth':3, 'bst:eta':1, 'silent':1, 'objective':'multi:softprob'}
param = {'bst:max_depth':3, 'bst:eta':.5, 'silent':4, 'objective':'multi:softprob'}
param['nthread'] = 8
param['num_class'] = 9 
plst = param.items()
plst += [('eval_metric', 'mlogloss')]   # Multiple evals can be handled in this way
#plst += [('eval_metric', 'auc')]        # Multiple evals can be handled in this way

evallist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 430

# train model
bst = xgb.train(plst, dtrain, num_round, evallist)
bst.save_model('0001.model')
bst.dump_model('dump.raw.txt') #, 'featuremap.txt')

# make predictions
testdata = xgb.DMatrix(data)
proba = bst.predict(testdata)
print("Generating submission...")
filename = "xgb-otto-proba-round-%d-eta-%d.csv"%(num_round, param['bst:eta'])
ut.make_submission(proba, ids, encoder, fname=filename)
