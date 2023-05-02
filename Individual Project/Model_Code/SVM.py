import pandas as pd
from sklearn import svm
# from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class SVM():
    def __init__(self, data, strategy):
        self.data = data
        self.x = data.drop(['evaluation'], axis = 1)
        self.y = data['evaluation']
        # strategy is a string to store
        # which strategy we will use to train model.
        # 'OVO' or 'OVR'
        self.strategy = strategy

    def encode_data(self):
        # use OneHotEncoder to encode data.
        enc = OneHotEncoder()
        enc.fit(self.x)
        x_encoder = pd.DataFrame(enc.transform(self.x).toarray())
        # data_encoder = pd.concat([x_encoder, self.y], axis = 1)
        return x_encoder

    def spilt_data(self, x_encoder):
        # use train_test_split to separate training set into
        # training data and validation data.
        x_train, x_valid, y_train, y_valid = train_test_split(x_encoder, self.y, test_size = 0.3, random_state = 22)
        return x_train, x_valid, y_train, y_valid

    def svm_OneVsOne_fit(self, x_train, y_train):
        clt = svm.SVC(kernel = 'poly', decision_function_shape = 'ovo')
        clt = clt.fit(x_train, y_train)
        return clt

    def svm_OneVsRest_fit(self, x_train, y_train):
        # here, we use one VS rest method to do classification.
        clt = OneVsRestClassifier(svm.SVC(kernel = 'poly', probability = True, random_state = 22))
        clt = clt.fit(x_train, y_train)
        # save model in pkl file,
        # we can use it directly after one-time training.
        # joblib.dump(clt,"model/conv_19_80%.pkl")
        return clt

    def predict(self, model, x_valid):
        y_valid_pred = model.predict(x_valid)
        return y_valid_pred

    def get_valid_y(self):
        return self.spilt_data(self.encode_data())[3]

    def get_result(self):
        print('-----------SVM Model-----------')
        x_e = self.encode_data()
        x_train, x_valid, y_train, y_valid = self.spilt_data(x_e)
        if self.strategy == 'OVO':
            print('-----------One VS One-----------')
            print()
            model = self.svm_OneVsOne_fit(x_train, y_train)
        if self.strategy == 'OVR':
            print('-----------One VS Rest-----------')
            print()
            model = self.svm_OneVsRest_fit(x_train, y_train)
        
        res = self.predict(model, x_valid)
        return res

# # k fold validation for SVM Model
# kf = KFold(n_splits=5)
# for k, (train_index, vad_index) in enumerate(kf.split(train_set)):
#     x_train, x_vad, y_train, y_vad = train_dummies.values[train_index], train_dummies.values[vad_index], train_target[
#         train_index], list(train_target[vad_index])
#     model = SVC(kernel='poly', C=1)
#     svm = model.fit(x_train, y_train)
#     y_pred = list(svm.predict(x_vad))
    
#     print('SVM {} fold'.format(k + 1))
#     # score(y_pred,y_vad)
