import numpy as np 
import re
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import svm
import seaborn as sns
import mydata as md
import utils as ut
import matplotlib.pyplot as plt
import pylab as pl
import os, sys,getopt,time, gc 


def evalError(model, traindata, testdata, plot=False):
    x_test, y_test   = testdata
    x_train, y_train = traindata
    
    if plot:
        nestim = len(model.estimators_)
        train_errors = np.empty(nestim)
        test_errors  = np.empty(nestim)
    
        for i, pred in enumerate(model.staged_predict(x_test)):
            test_errors[i] = sum(np.not_equal(y_test, pred)) 
        plt.plot(np.arange(nestim)+1, test_errors, label="Test")
    
        for i, pred in enumerate(model.staged_predict(x_train)):
            train_errors[i] = sum(np.not_equal(y_train, pred)) 

        plt.plot(np.arange(nestim)+1, train_errors, label="Train")
        plt.show()
    
    test_acc = model.score(x_test, y_test)
    train_acc = model.score(x_train, y_train)
    #print mean_squared_error(y_test, clf.predict(x_test))
    #predictions = model.predict(x_test)
    return [train_acc, test_acc]
    


def main(argv):
    learnModel = True   # by default we learn each model again 

    try:
        opts, args = getopt.getopt(argv,"hl:",["load="])
    except getopt.GetoptError:
        print 'analyticSuite.py -l <modelpath>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'analyticSuite.py -l <path>'
            sys.exit()
        elif opt in ("-l", "--load"):
            modelpath = arg
            learnModel = False    # do not learn, load an existing model
    
        
    ################################################
    # load_data returns data in tuple form (x,y)
    ################################################
    if not learnModel:
        unseen_data = md.load_data('data/test.csv', datatype="test", verbose=True)
    
   
    ################################################
    # GradientBoosting classifier
    ################################################
    if learnModel:
        # load training dataset and split         
        train,test,valid = md.load_data( 'data/train.csv', datatype="train", verbose=True) 
        x_train,y_train  = train
        x_test, y_test   = test 
        x_valid, y_valid = valid
        
        params = {'n_estimators': 800, 
                  'max_depth': 3, 
                  'subsample': 0.5,
                  'learning_rate': 0.1, 
                  'min_samples_leaf': 1}
        
        # train Gradient Boosting Classifier
        gbc = ensemble.GradientBoostingClassifier(**params)
        gbc.fit(x_train, y_train)
        train_acc, test_acc = evalError(gbc, (x_train,y_train), (x_test,y_test) )
        print "GradientBoostingClassifier\ntrain accuracy %f\ntest accuracy %f\n\n" %(train_acc, test_acc) 
        ut.save(gbc, "GradientBoosting", verbose=True)
        
    else:
        print "Loading Gradient Boosting model"
        gbc = ut.load("GradientBoosting", modelpath)
        
    #ut.writePredictionFile(gbc.predict(unseen_data), modeltype="gbc")
    ut.writePredictionFile(gbc.predict_proba(unseen_data), modeltype="gbc")

    # free memory
    del gbc
    gc.collect()
    
    ##############################################
    ## Support Vector Machine ## 
    ##############################################
    if learnModel:
        svmc = svm.SVC(probability=True)
        svmc.fit(x_train, y_train)
        ut.save(svmc, "SupportVectorMachine", verbose=True)
        train_acc, test_acc = evalError(svmc, (x_train,y_train), (x_test,y_test))
        print "ExtraTreeClassifier\ntrain accuracy %f\ntest accuracy %f\n\n" %(train_acc, test_acc)
        #svmscores = cross_val_score(svmc, x_train, y_train)
        #print "SupportVectorMachine mean score %f" %svmscores.mean() 
 
    else:
        print "Loading SVM model"
        svmc = ut.load("SupportVectorMachine", modelpath)
        
    ut.writePredictionFile(svmc.predict(unseen_data), modeltype="svm")
    # free memory
    del svmc
    gc.collect()
    
    ############################################
    # ExtraTreesClassifier 
    ############################################
   
    if learnModel:
        etc = ensemble.ExtraTreesClassifier(n_estimators=600, 
                                            max_depth=None,
                                            min_samples_split=1, 
                                            random_state=0, 
                                            n_jobs=-1)
        etc.fit(x_train, y_train)
        ut.save(etc, "ExtraTreesClassifier", verbose=True)
        train_acc, test_acc = evalError(etc, (x_train,y_train), (x_test,y_test))
        print "ExtraTreeClassifier\ntrain accuracy %f\ntest accuracy %f\n\n" %(train_acc, test_acc) 
        #scores = cross_val_score(etc, x_train, y_train)
        
    else:
        print "Loading ExtraTrees Classifier model"
        etc = ut.load("ExtraTreesClassifier", modelpath)
    
    #ut.writePredictionFile(etc.predict(unseen_data), modeltype="etc")
    ut.writePredictionFile(etc.predict_proba(unseen_data), modeltype="etc")
    
    # free memory
    del etc
    gc.collect()
   
    
   
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
