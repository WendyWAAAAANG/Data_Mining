'''
here is a multi-classify problem!!
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def Perceptron(x_train, y_train, x_valid, learning_rate, training_time):
    print('-------Perceptron-------')

    # use counter to node the time of iterations.
    counter = 0
    # encode y using 0 and 1,
    # we have to use two map for two perceptron.
    y_train_map1 = list(y_train.map({'good': 0, 'acc': 0, 'unacc': 1}))
    y_train_map2 = list(y_train.map({'good': 0, 'acc': 1}))
    # w and b initialization, using random number.
    w_1 = np.random.rand(21, )
    b_1 = 0
    w_2 = np.random.rand(21, )
    b_2 = 0
    # start training model.
    while(counter < training_time):
        # use a bool variable to find out whether model is converge.
        isConverge = 1
        for i in range(len(x_train) - 1): 
            # use w and b to fit model.
            pred = b_1 + (w_1 * (x_train.loc[i,:])).sum()
            if(pred > 0):
                if y_train_map1[i] != 1:
                    # which means the item in the wrong class.
                    # we need to update parameters.
                    # change isConverge to 1, because no converage
                    isConverge = 0
                    # update w and b.
                    w_1 = w_1 + learning_rate * (y_train_map1[i] - 1) * x_train.loc[i,:]
                    b_1 = b_1 + learning_rate * (y_train_map1[i] - 1)
            else:
                if y_train_map1[i] != 0:
                    # which means the item in the wrong class.
                    # we need to update parameters.
                    # change isConverge to 1, because no converage
                    isConverge = 0
                    # update w and b.
                    w_1 = w_1 + learning_rate * (y_train_map1[i]) * x_train.loc[i,:]
                    b_1 = b_1 + learning_rate * (y_train_map1[i])
                else:
                    pred_1 = b_2 + (w_2 * x_train.loc[i,:]).sum()
                    if pred_1 > 0:
                        if y_train_map2[i] != 1:
                            # which means the item in the wrong class.
                            # we need to update parameters.
                            # change isConverge to 1, because no converage
                            isConverge = 0
                            # update w and b.
                            b_2 = b_2 + learning_rate * (y_train_map2[i] - 1)
                            w_2 = w_2 + learning_rate * (y_train_map2[i] - 1) * x_train.loc[i,:]
                    else:
                        if y_train_map2[i] != 0:
                            # which means the item in the wrong class.
                            # we need to update parameters.
                            # change isConverge to 1, because no converage
                            isConverge = 0
                            # update w and b.
                            b_2 = b_2 + learning_rate * y_train_map2[i]
                            w_2 = w_2 + learning_rate * (y_train_map2[i]) * x_train.loc[i,:]
        if isConverge == 1:
            # which means the model has already converged.
            print('After {} iterations, it has converged!'.format(counter))
            break
        # print out message of process.
        counter += 1
        print('{}/{} finished'.format(counter, training_time))
        
    # Get the predict result
    pred_res = []
    # start to predict for all item in testset.
    for i in range(len(x_valid)):
        # use current w_1 and b_1 to predict result.
        pred = b_1 + (w_1 * x_valid.loc[i,:]).sum()
        # check which class the result belong to.
        if pred > 0:
            # which means it is unacc.
            pred_res.append('unacc')
        else:
            # otherwise, use w_2 and b_2 to predict whether good or acc.
            pred_1 = b_2 + (w_2 * x_valid.loc[i,:]).sum()
            if pred_1 < 0:
                # which means the result is good.
                pred_res.append('good')
            else:
                # otherwise, the predict result is acc.
                pred_res.append('acc')
    print('Done')
    return pred_res

# pred_res = Perceptron(x_train.reset_index(drop=True),y_train.reset_index(drop=True),x_valid.reset_index(drop=True),0.2, 50)



# import pandas as pd
# import numpy as np
# from math import *
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing

# class Perceptron():
#     def __init__(self, data, lr, training_time):
#         self.data = data
#         self.x_train = pd.DataFrame()
#         self.y_train = pd.DataFrame()
#         # lr -- learning rate.
#         self.lr = lr
#         self.w = np.random.randn(3, 6)
#         self.b = np.random.randn(3)
#         self.training_time = training_time
#         self.alpha = [[0] * 6, [0] * 6, [0] * 6]
#         # self.beta = []
#         self.beta0 = 0; self.beta1 = 0; self.beta2 = 0
#         self.loss = 0
#         self.labels = [-1, 0, 1]
#         # classMap used to transform y into vector.
#         self.classMap = {'-1': [1, 0, 0],
#                         '0': [0, 1, 0],
#                         '1': [0, 0, 1]}

#     def preprocess_data(self):
#         x = self.data.drop(['evaluation'], axis = 1)
#         y = self.data['evaluation']
#         # since all variables in x has its sequence,
#         # encoder it according to their sequence.
#         le = preprocessing.LabelEncoder()
#         for col in x:
#             x_tran = le.fit_transform(data[col].tolist())
#             tran_df = pd.DataFrame(x_tran, columns=['num_' + col])
#             # print('{col} has transformed into {num_col}'.format(col = col, num_col = 'num_' + col))
#             x = pd.concat([x, tran_df], axis = 1)
#             # delete pervious columns.
#             del x[col]
#         # use dummy variables to represent y.
#         y_encode = y
#         y_encode[y_encode == 'acc'] = 1
#         y_encode[y_encode == 'unacc'] = 0
#         y_encode[y_encode == 'good'] = -1
#         data_encode = pd.concat([x, y_encode], axis = 1)
#         return data_encode, y_encode
        
#     def spilt_data(self, data_encode, y_encode):
#         # use train_test_split to separate training set into
#         # training data and validation data.
#         self.x_train, x_valid, self.y_train, y_valid = train_test_split(data_encode.iloc[:, :-1], y_encode, test_size = 0.3, random_state = 22)
#         self.x_train = self.x_train.to_numpy()
#         self.y_train = self.y_train.to_numpy()
#         x_valid = x_valid.to_numpy()
#         y_valid = y_valid.to_numpy()
#         # self.data_train = pd.concat([x_train, y_train], axis = 1)
#         # # convert y_valid into list.
#         # y_valid = list(y_valid)
#         return x_valid, y_valid

#     def update_para(self):
#         self.w[0] -= self.alpha[0] * self.lr
#         self.w[1] -= self.alpha[1] * self.lr
#         self.w[2] -= self.alpha[2] * self.lr
#         self.b[0] -= self.beta0 * self.lr
#         self.b[1] -= self.beta1 * self.lr
#         self.b[2] -= self.beta2 * self.lr
#         self.loss = self.loss / len(self.x_train)
        
#     def calculate_loss(self):
#         for i, j in zip(self.x_train, self.y_train):
#             # calculate output using hidden layer weight and b.
#             z = np.sum(np.multiply([i] * 3, self.w), axis = 1) + self.b
#             # here, we use softmax function, for multiclassify.
#             # calculate output, softmax(z).
#             y_predict = np.exp(z) / sum(np.exp(z))
#             # fetch the y vector.
#             y_i = self.classMap[str(j)]
#             # calculate loss function for current sample.
#             # here is 交叉熵 loss function.
#             lossi = -sum(np.multiply(y_i, np.log(y_predict)))
#             # add up loss.
#             self.loss += lossi

#             # use partical derivative to update weight.
#             self.alpha[0] += np.multiply(sum(np.multiply([0, 1, 1], y_i)), i)
#             self.alpha[1] += np.multiply(sum(np.multiply([1, 0, 1], y_i)), i)
#             self.alpha[2] += np.multiply(sum(np.multiply([1, 1, 0], y_i)), i)
#             self.beta0 += sum(np.multiply([0, 1, 1], y_i))
#             self.beta1 += sum(np.multiply([1, 0, 1], y_i))
#             self.beta2 += sum(np.multiply([1, 1, 0], y_i))

#     # def prediction(self):

#     def get_result(self):
#         class_map = [-1, 0, 1]
#         recall=0
#         # initialize result list.
#         result = []
#         # here, training model.
#         d_e, y_e = self.preprocess_data()
#         x_v, y_v = self.spilt_data(d_e, y_e)
#         for i in range(self.training_time):
#             self.calculate_loss()
#             self.update_para()
#             # here, use model already trained perdict.
#             for k, _ in zip(x_v, y_v):
#                 ai = np.sum(np.multiply([k] * 3, self.w), axis = 1) + self.b
#                 y_predicti = np.exp(ai) / sum(np.exp(ai))
#                 y_predicti = [class_map[idx] for idx, i in enumerate(y_predicti) if i == max(y_predicti)][0]
#                 result.append(y_predicti)
#                 recall += 1 if int(y_predicti) == int(i) else 0
#         # Fit the Perceptron model and use it to predict result.
#         print('--------Perceptron Model--------')
#         print('验证集总条数：', len(x_v), '预测正确数：', recall)
#         res_df = pd.DataFrame(result)
#         # print(res_df)
#         return res_df

# data = pd.read_csv('Dataset/training.csv')
# t = Perceptron(data, 0.01, 5).get_result()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def Perceptron(x_train, y_train, x_valid, learning_rate, training_time):
    print('-------Perceptron-------')

    # use counter to node the time of iterations.
    counter = 0
    # encode y using 0 and 1,
    # we have to use two map for two perceptron.
    y_train_map1 = list(y_train.map({'good': 0, 'acc': 0, 'unacc': 1}))
    y_train_map2 = list(y_train.map({'good': 0, 'acc': 1}))
    # w and b initialization, using random number.
    w_1 = np.random.rand(21, )
    b_1 = 0
    w_2 = np.random.rand(21, )
    b_2 = 0
    # start training model.
    while(counter < training_time):
        # use a bool variable to find out whether model is converge.
        isConverge = 1
        for i in range(len(x_train) - 1): 
            # use w and b to fit model.
            pred = b_1 + (w_1 * (x_train.loc[i,:])).sum()
            if(pred > 0):
                if y_train_map1[i] != 1:
                    # which means the item in the wrong class.
                    # we need to update parameters.
                    # change isConverge to 1, because no converage
                    isConverge = 0
                    # update w and b.
                    w_1 = w_1 + learning_rate * (y_train_map1[i] - 1) * x_train.loc[i,:]
                    b_1 = b_1 + learning_rate * (y_train_map1[i] - 1)
            else:
                if y_train_map1[i] != 0:
                    # which means the item in the wrong class.
                    # we need to update parameters.
                    # change isConverge to 1, because no converage
                    isConverge = 0
                    # update w and b.
                    w_1 = w_1 + learning_rate * (y_train_map1[i]) * x_train.loc[i,:]
                    b_1 = b_1 + learning_rate * (y_train_map1[i])
                else:
                    pred_1 = b_2 + (w_2 * x_train.loc[i,:]).sum()
                    if pred_1 > 0:
                        if y_train_map2[i] != 1:
                            # which means the item in the wrong class.
                            # we need to update parameters.
                            # change isConverge to 1, because no converage
                            isConverge = 0
                            # update w and b.
                            b_2 = b_2 + learning_rate * (y_train_map2[i] - 1)
                            w_2 = w_2 + learning_rate * (y_train_map2[i] - 1) * x_train.loc[i,:]
                    else:
                        if y_train_map2[i] != 0:
                            # which means the item in the wrong class.
                            # we need to update parameters.
                            # change isConverge to 1, because no converage
                            isConverge = 0
                            # update w and b.
                            b_2 = b_2 + learning_rate * y_train_map2[i]
                            w_2 = w_2 + learning_rate * (y_train_map2[i]) * x_train.loc[i,:]
        if isConverge == 1:
            # which means the model has already converged.
            print('After {} iterations, it has converged!'.format(counter))
            break
        # print out message of process.
        counter += 1
        print('{}/{} finished'.format(counter, training_time))
        
    # Get the predict result
    pred_res = []
    # start to predict for all item in testset.
    for i in range(len(x_valid)):
        # use current w_1 and b_1 to predict result.
        pred = b_1 + (w_1 * x_valid.loc[i,:]).sum()
        # check which class the result belong to.
        if pred > 0:
            # which means it is unacc.
            pred_res.append('unacc')
        else:
            # otherwise, use w_2 and b_2 to predict whether good or acc.
            pred_1 = b_2 + (w_2 * x_valid.loc[i,:]).sum()
            if pred_1 < 0:
                # which means the result is good.
                pred_res.append('good')
            else:
                # otherwise, the predict result is acc.
                pred_res.append('acc')
    print('Done')
    return pred_res


# # data pre-processing.
# # import training data, separate into x and y.
# data = pd.read_csv('Dataset/training.csv')
# x = data.drop(['evaluation'], axis = 1)
# y = data['evaluation']

# # use OneHotEncoder to encode data.
# enc = OneHotEncoder()
# enc.fit(x)
# x_encoder = pd.DataFrame(enc.transform(x).toarray())

# # use train_test_split to separate training set into training data and validation data.
# x_train, x_valid, y_train, y_valid = train_test_split(x_encoder, y, test_size = 0.3, random_state = 22)

# pred_res = Perceptron(x_train.reset_index(drop=True),y_train.reset_index(drop=True),x_valid.reset_index(drop=True),0.2, 50)