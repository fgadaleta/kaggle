import numpy as np 
import re
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
import mydata as md
import seaborn as sns
from pandas import DataFrame



train,test = md.load_data() 
x_train = train[0]
y_train = train[1]

numsamples= len(x_train)
trainlen = np.ceil(0.70*numsamples)
testlen  = np.ceil(0.25*numsamples)

# shuffle the indices not the data!
data_idx = np.arange(numsamples)
np.random.shuffle(data_idx)    
train_idx = data_idx[:trainlen]

x_train_set = x_train[train_idx,]
y_train_set = y_train[train_idx]


df = DataFrame(np.hstack((x_train_set, y_train_set[:, None])))


plt.figure(figsize=(12, 10))
_ = sns.corrplot(df[:5000], annot=False)

plt.show()
