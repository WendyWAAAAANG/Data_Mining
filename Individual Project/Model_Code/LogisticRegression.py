# ligistic regression for multi class with One VS Rest.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

class LogRegression():
    def __init__(self, data, solver, max_iter, multi_class, class_weight):
        self.data = data
        self.x = data.drop('evaluation', axis = 1)
        self.y = data['evaluation']
        # initialize for further split data.
        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_valid = pd.DataFrame()
        self.y_valid = pd.DataFrame()
        # store some parameter of model.
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.class_weight = class_weight
    
    def encode_data(self):
        # use OneHotEncoder to encode data.
        enc = OneHotEncoder()
        enc.fit(self.x)
        self.x = pd.DataFrame(enc.transform(self.x).toarray())

    def spilt_data(self):
        self.x_train, self.x_valid, self.y_train, y_valid = train_test_split(self.x,\
                  self.y, test_size = 0.3, random_state = 22)
        return y_valid
    
    def fit_model(self):
        clf = LogisticRegression(solver = self.solver, max_iter = self.max_iter,\
                multi_class = self.multi_class, class_weight = self.class_weight)
        clf.fit(self.x_train, self.y_train)
        return clf
    
    def prediction(self, clf):
        res = clf.predict(self.x_valid)
        return res
    
    def get_result(self):
        print('-------Logistic_Regression Model-------')
        self.encode_data()
        self.spilt_data()
        clf = self.fit_model()
        res = self.prediction(clf)
        print('Done')
        return res



# def LogRegre (x, y, solver, max_iter, multi_class, class_weight):
#     x_train, x_valid, y_train, _ = train_test_split(x, y, test_size = 0.3, random_state = 22)

#     clf = LogisticRegression(random_state = 22, solver = solver,\
#             max_iter = max_iter, multi_class = multi_class, class_weight = class_weight)
#     clf.fit(x_train, y_train)

#     y_pred = clf.predict(x_valid)
#     return y_pred