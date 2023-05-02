from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt

class DecisionTree():
    def __init__(self, x_train, x_valid, y_train, y_valid):
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def encodeData(self):
        # encode variables with ordinal encoding
        encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
        self.x_train = encoder.fit_transform(self.x_train)
        self.x_valid = encoder.transform(self.x_valid)

    def fitModel(self):
        clf_gini = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 22)
        # fit the model
        clf_gini.fit(self.x_train, self.y_train)
        return clf_gini
        
    def checkOverfitting(self, clf_gini):
        # check for overfitting.
        print('Training set score: {:.4f}'.format(clf_gini.score(self.x_train, self.y_train)))
        print('Test set score: {:.4f}'.format(clf_gini.score(self.x_valid, self.y_valid)))

    def DTVisualize(self, clf_gini):
        # visualize DT.
        plt.figure(figsize = (12, 8))
        tree.plot_tree(clf_gini.fit(self.x_train, self.y_train)) 
        #Visualize decision-trees with graphviz
        dot_data = tree.export_graphviz(clf_gini, out_file = None, 
                                    feature_names = self.x_train.columns,  
                                    class_names = self.y_train,  
                                    filled = True, rounded = True,  
                                    special_characters = True)
        graph = graphviz.Source(dot_data)
    
    def prediction(self, clf_gini):
        res_DT = clf_gini.predict(self.x_valid)
        res_DT[0:5]
        return res_DT

    def get_result(self):
        print('-------Decision_Tree Model-------')
        self.encodeData()
        clf = self.fitModel()
        self.checkOverfitting(clf)
        self.DTVisualize(clf)
        res_DT = self.prediction(clf)
        print('Done')
        return np.array(res_DT)


    

# # use train_test_split to separate training set into
# # training data and validation data.
# from sklearn.model_selection import train_test_split
# x_train, x_valid, y_train, y_valid = train_test_split(data.iloc[:, :-1], y, test_size = 0.3, random_state = 22)
# data_train = pd.concat([x_train, y_train], axis = 1)

# # encode variables with ordinal encoding
# encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
# X_train = encoder.fit_transform(x_train)
# X_test = encoder.transform(x_valid)

# clf_gini = DecisionTreeClassifier(criterion='gini', max_depth = 3, random_state = 22)
# # fit the model
# clf_gini.fit(x_train, y_train)
# y_pred_gini = clf_gini.predict(x_valid)
# y_pred_gini[0:5]


# # check for overfitting.
# print('Training set score: {:.4f}'.format(clf_gini.score(x_train, y_train)))
# print('Test set score: {:.4f}'.format(clf_gini.score(x_valid, y_valid)))


# # visualize DT.
# plt.figure(figsize=(12,8))
# tree.plot_tree(clf_gini.fit(x_train, y_train)) 
# #Visualize decision-trees with graphviz
# dot_data = tree.export_graphviz(clf_gini, out_file=None, 
#                               feature_names = x_train.columns,  
#                               class_names=y_train,  
#                               filled = True, rounded=True,  
#                               special_characters=True)
# graph = graphviz.Source(dot_data) 
# graph 



          


# if __name__ == '__main__':
#     fr = open('play.tennies.txt')
#     lenses =[inst.strip().split(' ') for inst in fr.readlines()]
#     lensesLabels = ['outlook','temperature','huminidy','windy']
#     lensesTree =createTree(lenses,lensesLabels)
#     treePlotter.createPlot(lensesTree)


# #!/usr/bin/python
# #encoding:utf-8
# #treePlotter.py
# import matplotlib.pyplot as plt

# decisionNode = dict(boxstyle="sawtooth", fc="0.8") #定义文本框与箭头的格式
# leafNode = dict(boxstyle="round4", fc="0.8")
# arrow_args = dict(arrowstyle="<-")

# def getNumLeafs(myTree): #获取树叶节点的数目
#     numLeafs = 0
#     firstStr = myTree.keys()[0]
#     secondDict = myTree[firstStr]
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=='dict':#测试节点的数据类型是不是字典，如果是则就需要递归的调用getNumLeafs()函数
#             numLeafs += getNumLeafs(secondDict[key])
#         else:   numLeafs +=1
#     return numLeafs

# def getTreeDepth(myTree): #获取树的深度
#     maxDepth = 0
#     firstStr = myTree.keys()[0]
#     secondDict = myTree[firstStr]
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
#             thisDepth = 1 + getTreeDepth(secondDict[key])
#         else:   thisDepth = 1
#         if thisDepth > maxDepth: maxDepth = thisDepth
#     return maxDepth

# # 绘制带箭头的注释
# def plotNode(nodeTxt, centerPt, parentPt, nodeType):
#     createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
#              xytext=centerPt, textcoords='axes fraction',
#              va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

# #计算父节点和子节点的中间位置，在父节点间填充文本的信息
# def plotMidText(cntrPt, parentPt, txtString):




# import pandas as pd
# from math import log
# import operator
# from sklearn.model_selection import train_test_split


# data = pd.read_csv('Dataset/training.csv')

# class DecisionTree():

#     def __init__(self, data):
#         self.data = data
#         self.x = data.drop(['evaluation'], axis = 1)
#         self.y = data['evaluation']

#         self.x_train, self.x_valid, self.y_train, y_valid = train_test_split(self.x,\
#              self.y, test_size = 0.3, random_state = 22)
#         self.data_train = pd.concat([self.x_train, self.y_train], axis = 1)
#         # store featue of x into feature_list.
#         self.labels = self.data.columns.tolist()[ :-1]

#     # def createDataSet(self):
#     #     dataSet = [[1, 1, 'yes'],
#     #             [1, 1, 'yes'],
#     #             [1, 0, 'no'],
#     #             [0, 1, 'no'],
#     #             [0, 1, 'no']]
#     #     labels = ['no surfacing','flippers']
#     #     #change to discrete values
#     #     return dataSet, labels

#     def cal_ShannonEnt(self):
#         # used to store length of data.
#         data_num = len(self.data_train)
#         # used to store the frequency of data.
#         label_num = {}
#         # calculate the number of unique elements and their occurance.
#         for feature in self.data_train:
#             current_label = feature[-1]
#             if current_label not in label_num.keys():
#                 # which means there is no current label.
#                 # set its number to 0.
#                 label_num[current_label] = 0
#             # otherwise, increase the number of count.
#             label_num[current_label] += 1
#         # shannon entropy initialization.
#         shannonEnt = 0.0
#         # use formula of en
#         for key in label_num:
#             # calculate each probability.
#             prob = float(label_num[key]) / data_num
#             # take log in base 2.
#             shannonEnt -= prob * log(prob,2)
#         return shannonEnt
    
#     # find the proper feature to spilt data.
#     # use feature th feature which value is value to spilt.
#     def reduce_data(self, feature, value):
#         reduced_data = []
#         for item in self.data_train:
#             if item[feature] == value:
#                 # chop out feature used for splitting
#                 reduced_feature = item[ :feature]
#                 reduced_feature.extend(item[feature + 1 :])
#                 reduced_data.append(reduced_feature)
#         return reduced_data
        
#     def chooseBestFeatureToSplit(self):
#         # the last column is used for the labels.
#         feature_num = len(self.data_train[0]) - 1
#         baseEntropy = self.cal_ShannonEnt(self.data_train)
#         # initialize info gain.
#         bestInfoGain = 0.0
#         bestFeature = -1
#         # iterate over all the features.
#         for i in range(feature_num):
#             # create a list of all the examples of this feature.
#             featList = [example[i] for example in self.data_train]
#             # get a set of unique values.
#             uniqueVals = set(featList)
#             newEntropy = 0.0
#             for value in uniqueVals:
#                 subDataSet = self.reduce_data(self.data_train, i, value)
#                 prob = len(subDataSet) / float(len(self.data_train))
#                 newEntropy += prob * self.cal_ShannonEnt(subDataSet)
#             # calculate the info gain; ie reduction in entropy.
#             infoGain = baseEntropy - newEntropy
#             # compare this to the best gain so far.
#             if (infoGain > bestInfoGain):
#                 # if better than current best, set to best.
#                 bestInfoGain = infoGain
#                 bestFeature = i
#         return bestFeature

#     def majorityCnt(self):
#         # initialize class_count, to 
#         class_count = {}
#         for vote in self.y_train:
#             if vote not in class_count.keys():
#                 class_count[vote] = 0
#             class_count[vote] += 1
#         # here we use operator to use keys' value sorting dict.
#         sortedClassCount = sorted(class_count.iteritems(),\
#                 key = operator.itemgetter(1), reverse = True)
#         # return the name of class which apprears most frequently.
#         return sortedClassCount[0][0]

#     def create_tree(self):
#         # classList = [example[-1] for example in self.data_train]
#         # if classList.count(classList[0]) == len(classList):
#         #     # stop splitting when all of the classes are equal.
#         #     return classList[0]
#         # # stop splitting when there are no more features in dataSet.
#         # if len(self.data_train[0]) == 1:
#         #     return self.majorityCnt(classList)

#         if self.y_train.count(self.y_train[0]) == len(self.y_train):
#             # stop splitting when all of the classes are equal.
#             return self.y_train[0]
        
#         # stop splitting when there are no more features in dataSet.
#         if len(self.data_train[0]) == 1:
#             return self.majorityCnt(self.y_train)

#         # find the spilt criteria using chooseBestFeatureToSpilt function.
#         best_feature_index = self.chooseBestFeatureToSplit(self.data_train)
#         # fetch value of the bestFeature.
#         best_feature_label = self.labels[best_feature_index]
#         # use dict to store information of decision tree,
#         # for further graphing DT.
#         DT = {best_feature_label: {}}
#         del(self.labels[best_feature_index])
#         feature_value = [example[best_feature_index] for example in self.data_train]
#         # use a set to store 
#         uniqueVals = set(feature_value)
#         for value in uniqueVals:
#             # copy all of labels, so trees don't mess up existing labels.
#             subLabels = self.labels[ : ]
#             DT[best_feature_label][value] = self.create_tree\
#                     (self.reduce_data(self.data_train, best_feature_index, value), subLabels)
#         return DT                            
        
#     def classify(self, inputTree, featLabels, testVec):
#         firstStr = inputTree.keys()[0]
#         secondDict = inputTree[firstStr]
#         featIndex = featLabels.index(firstStr)
#         key = testVec[featIndex]
#         valueOfFeat = secondDict[key]
#         if isinstance(valueOfFeat, dict): 
#             classLabel = self.classify(valueOfFeat, featLabels, testVec)
#         else:
#             classLabel = valueOfFeat
#         return classLabel

#     def store_tree(self, inputTree, filename):
#         import pickle
#         fw = open(filename,'w')
#         pickle.dump(inputTree, fw)
#         fw.close()
        
#     def grab_tree(self, filename):
#         import pickle
#         fr = open(filename)
#         return pickle.load(fr)
    
