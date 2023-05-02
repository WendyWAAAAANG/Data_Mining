import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import category_encoders as ce
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class XGBoost():
    def __init__(self, data):
        self.data = data
        self.x = data.drop(['evaluation'], axis = 1)
        self.y = data['evaluation']
        self.x_train = pd.DataFrame()
        self.x_valid = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_valid = pd.DataFrame()
        
    def spilt_data(self):
        # use train_test_split to separate training set into
        # training data and validation data.
        self.x_train, self.x_valid, self.y_train, self.y_valid = \
                train_test_split(self.x, self.y, test_size = 0.3, random_state = 22)
        return self.y_valid
        
    def encode_data(self):
        # encode y using [0 1 2].
        self.y_train[self.y_train == 'acc'] = 0
        self.y_train[self.y_train == 'unacc'] = 1
        self.y_train[self.y_train == 'good'] = 2

        self.y_valid[self.y_valid == 'acc'] = 0
        self.y_valid[self.y_valid == 'unacc'] = 1
        self.y_valid[self.y_valid == 'good'] = 2

        # Encode training features with ordinal encoding.
        encoder = ce.OrdinalEncoder(cols = self.x_train)
        self.x_train = encoder.fit_transform(self.x_train)
        self.x_valid = encoder.transform(self.x_valid)
        
    def get_valid_y(self):
        return self.y_valid
                
    def fit_model(self):
        clf_xgb = XGBClassifier(eval_metric = 'mlogloss', n_estimators = 200, max_features = 5)
        clf_xgb.fit(self.x_train, self.y_train)
        return clf_xgb
    
    def prediction(self, clf_xgb):
        # Predict and test on test data
        result = pd.DataFrame(clf_xgb.predict(self.x_valid))
        return result
                        
    def get_result(self):
        print('--------XGBoost Model--------')
        self.spilt_data()
        self.encode_data()
        clf = self.fit_model()
        res = self.prediction(clf)
        res[res == 0] = 'acc'
        res[res == 1] = 'unacc'
        res[res == 2] = 'good'
        print('Done')
        return res.values.reshape(399, )

# # data pre-processing.
# # import training data, separate into x and y.
# data = pd.read_csv('Dataset/training.csv')
# x = data.drop(['evaluation'], axis = 1)
# y = data['evaluation']
# # encode y using [0 1 2].
# y_encode = y
# y_encode[y_encode == 'acc'] = 0
# y_encode[y_encode == 'unacc'] = 1
# y_encode[y_encode == 'good'] = 2

# # use train_test_split to separate training set into
# # training data and validation data.
# from sklearn.model_selection import train_test_split
# x_train, x_valid, y_train, y_valid = train_test_split(x, y_encode, test_size = 0.3, random_state = 22) 

# # Encode training features with ordinal encoding.
# encoder = ce.OrdinalEncoder(cols = x_train)
# x_train = encoder.fit_transform(x_train)
# x_valid = encoder.transform(x_valid)

# xgb = XGBClassifier(eval_metric = 'mlogloss')
# xgb.fit(x_train, y_train)

# # Predict and test on test data
# xgb_y_hat = xgb.predict(x_valid)
# # accuracy_score(y_valid, xgb_y_hat)
# print(xgb_y_hat)


    