import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Bayes():
    def __init__(self, data):
        # self.test = test
        self.x = data.drop(['evaluation'], axis = 1)
        self.y = data['evaluation']
        self.data_train = pd.DataFrame()
        self.x_train = pd.DataFrame()
        self.x_valid = pd.DataFrame()

        # store the frequency of each label in to label_freq.
        # store the frequency of each label in to label_freq.
        self.label_freq = []
        self.prior_prob = {}

        # store featue of x into feature_list.
        self.feature_list = pd.DataFrame()

        # features used to let number of feature label correspond to value_list.
        self.feature_num_list = []
        # value_list used to store all values of each feature.
        self.value_list = []

        # initialize conditional_prob, further store conditional probability.
        self.conditional_prob = pd.DataFrame()

    def spilt_data(self):
        self.x_train, self.x_valid, y_train, y_valid = train_test_split(self.x,\
             self.y, test_size = 0.3, random_state = 22)
        self.data_train = pd.concat([self.x_train, y_train], axis = 1)
        self.label_freq = y_train.value_counts()
        # store featue of x into feature_list.
        self.feature_list = self.data_train.columns.tolist()[ :-1]
        return y_valid
        
    def get_feature_val_list(self):
        # for all feature of x.
        for i in self.feature_list:
            self.feature_num_list.extend(len(self.x_train[i].unique()) * [i])
            self.value_list.extend(self.x_train[i].unique().tolist())
            
    def cal_prior_prob(self):
        # calculate the prior probability.
        # P(Ci) = |Ci| / |D|.
        for i in self.label_freq.index:
            self.prior_prob[i] = self.label_freq[i] / sum(self.label_freq)

    def cal_condition_prob(self):
        # Calculate the conditional probability.
        # P(x|Ci) = P(x & Ci) / P(Ci).
        self.conditional_prob = pd.DataFrame(np.zeros((len(self.value_list), 3)),
                                columns = self.label_freq.index.tolist(),
                                index = [self.feature_num_list, self.value_list])
        # using P(Ci) calculated above.
        for i in self.label_freq.index:
            sub_df1 = self.data_train[self.data_train.iloc[ :, -1] == i]
            for j in self.feature_list:
                sub_df2 = sub_df1[j].value_counts()
                for k in self.data_train[j].value_counts().index.tolist():
                    if k not in sub_df2.index.tolist():
                        sub_df2[k] = 0
                    # here, we add Laplacian Correction to avoid error.
                    self.conditional_prob.loc[(j, k)].loc[i] = (sub_df2[k] + 1)\
                         / (sum(sub_df2) + len(sub_df2.index.tolist()))

    def prediction(self):
        # Predict the result by calculating the probability
        total_prob = {}
        pred_res = []
        # for each item in test data.
        for i in range(len(self.x_valid)):
            # for each feature of one item.
            for j in self.label_freq.index:
                # initialize probability of likelihood and posterior.
                likelihood, posterior_prob = 1, 1
                for k in self.feature_list:
                    likelihood *= self.conditional_prob[j][k][self.x_valid.iloc[i, :][k]]
                # calculate posterior probability, using likelihood and prior_prob.
                posterior_prob = self.prior_prob[j] * likelihood
                # add posterior result into dictionary, for further max selection.
                total_prob[j] = posterior_prob
            # find the maximum prob of each sample.
            pred_res.append(max(total_prob, key = total_prob.get))
        return pred_res

    def get_result(self):
        # Fit the Bayes model and use it to predict result.
        print('-----------Bayes Model-----------')
        self.spilt_data()
        self.get_feature_val_list()
        self.cal_prior_prob()
        self.cal_condition_prob()
        res = np.array(self.prediction())
        print('Done')
        return res