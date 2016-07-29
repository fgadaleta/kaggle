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

print("Preprocessing labels")
y, encoder = ut.preprocess_labels(labels)

prediction_files = ["xgb-otto-proba-round-430-eta-0.csv", 
                    "keras-otto-proba-93.csv"]
ensemble = ut.ensemble(prediction_files, weights=[0.4, 0.6]) 
ut.make_submission(ensemble, ids, encoder, fname='ensemble-otto-selected-93.csv')
