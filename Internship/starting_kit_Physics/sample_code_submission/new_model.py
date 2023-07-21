#from curses import raw
import pickle
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB
from DANN_program.DANN import build_DANN, compile_DANN, build_datasets_for_DANN, fit_DANN
import math


MODEL_CONSTANT = "Constant"
MODEL_NB = "NB"


PREPROCESS_TRANSLATION = "translation"


# ------------------------------
# Baseline Model
# ------------------------------
class Model:

    def __init__(self,
                 model_name=MODEL_NB,
                 X_train=None,
                 Y_train=None,
                 X_test=None,
                 preprocessing=False,
                 preprocessing_method=PREPROCESS_TRANSLATION,
                 case=None
                 ):

        self.model_name = model_name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.preprocessing = preprocessing
        self.preprocessing_method = preprocessing_method

        if case is None:
            self.case = case
        else:
            self.case = case - 1

        self._set_model()

    def _set_model(self):

        if self.model_name == MODEL_CONSTANT:
            self.clf = None
        if self.model_name == MODEL_NB:
            self.clf = GaussianNB()

        self.is_trained = False

    def _preprocess_translation(self, X):

        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        translation = test_mean - train_mean
        return (X - translation)

    def fit(self, X=None, y=None):

        if self.model_name != MODEL_CONSTANT:

            if X is None:
                X = self.X_train
            if y is None:
                y = self.Y_train

            self.clf.fit(X, y)
            self.is_trained = True

    def predict(self, X=None, preprocess=True, theta=0):

        if X is None:
            X = self.X_test

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])

        if self.preprocessing & preprocess:
            if self.preprocessing_method == PREPROCESS_TRANSLATION:
                X = self._preprocess_translation(X)

        # if decision function  > theta --> class 1
        # else --> class 0
        predictions = np.zeros(X.shape[0])
        decisions = self.decision_function(X)

        predictions = (decisions > theta).astype(int)
        return predictions

    def decision_function(self, X=None, preprocess=True):

        if X is None:
            X = self.X_test

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])

        if self.preprocessing and preprocess:
            if self.preprocessing_method == PREPROCESS_TRANSLATION:
                X = self._preprocess_translation(X)

        if self.model_name in [MODEL_NB]:
            predicted_score = self.clf.predict_proba(X)
            # Transform with log
            epsilon = np.finfo(float).eps
            predicted_score = -np.log((1/(predicted_score+epsilon))-1)
            decisions = predicted_score[:, 1]
        else:
            decisions = self.clf.decision_function(X)

        # decision function = decision function - theta
        # return decisions - self.theta
        return decisions

    def save(self, name):
        pickle.dump(self.clf, open(name + '.pickle', "wb"))

    def load(self, name):
        modelfile = name + '.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self


# ------------------------------
# DANN Model
# ------------------------------

class DANN :
    def __init__(self,
                 model_name="DANN_1",
                 input_size = (2,),
                 hp_lambda = 1.0,
                 class_weights=[1,1],
                 X_train=None,
                 Y_train=None,
                 X_test=None,
                 case=None
                 ):
        # Instantiate DANN
        self.model_name = model_name
        self.input_size = input_size
        self.hp_lambda = hp_lambda
        self.model = build_DANN(model_name,input_size,hp_lambda)

        # Compile DANN
        self.class_weights = class_weights
        compile_DANN(self.model,class_weights)
        self.is_trained = False

        # Define other attributes
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        if case is None:
            self.case = case
        else:
            self.case = case - 1

    def restructure_data (self, num_epochs, batch_size, X_train=None, Y_train=None, X_test=None, Y_test=None) :
        if X_train is None:
            self.X_train = self.X_train
        if Y_train is None:
            self.Y_train = self.Y_train
        if X_test is None:
            self.X_test = self.X_test
        self.Y_test = Y_test
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        training_dataset,testing_dataset = build_datasets_for_DANN ([self.X_train],[self.Y_train],[self.X_test],[self.Y_test],num_epochs,batch_size)
        self.training_dataset,self.testing_dataset = training_dataset[0],testing_dataset[0]

    def fit(self, num_epochs=None, batch_size=None):
        if num_epochs is None:
            try :
                self.num_epochs = self.num_epochs
            except :
                print("You need to call restructure_data before fitting")
        if batch_size is None:
            try :
                self.batch_size = self.batch_size
            except :
                print("You need to call restructure_data before fitting")
        
        num_source_samples = self.X_train.shape[0]
        num_target_samples = self.X_test.shape[0]
        train_steps_per_epoch = int((num_source_samples+num_target_samples)/self.batch_size)
        test_steps_per_epoch = int(num_target_samples/self.batch_size)

        history = fit_DANN(self.model,self.training_dataset,self.testing_dataset,self.num_epochs,train_steps_per_epoch,test_steps_per_epoch)

        self.is_trained = True

        return history

    def predict(self, X=None, theta=0):
        if X is None:
            X = self.X_test
        predictions = np.zeros(X.shape[0])
        decisions = self.decision_function(X)

        predictions = (decisions > theta).astype(int)

        return predictions

    def predict_probas(self, X=None):
        if X is None:
            X = self.X_test

        raw_prediction = self.model.predict(X)
        return raw_prediction
    
    def decision_function(self, X=None, preprocess=True):
        if X is None:
            X = self.X_test

        predicted_score = self.model.predict(X)[0][:,1]
        # Transform with log
        epsilon = np.finfo(float).eps
        predicted_score = -np.log((1/(predicted_score+epsilon))-1)
        decisions = predicted_score

        return decisions

    # def save(self, name):
    #     pickle.dump(self.model, open(name + '.pickle', "wb"))

    # def load(self, name):
    #     modelfile = name + '.pickle'
    #     if isfile(modelfile):
    #         with open(modelfile, 'rb') as f:
    #             self = pickle.load(f)
    #         print("Model reloaded from: " + modelfile)
    #     return self