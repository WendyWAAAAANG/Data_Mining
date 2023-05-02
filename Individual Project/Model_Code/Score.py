# this is used to calculate the criteria of each model.
# including accuracy, recall, F-score and preciption.
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

class Score():
    def __init__(self, actual_result, pred_result):
        # pass actual and predicted result to class.
        self.actual_result = actual_result
        self.pred_result = pred_result
        # calculate confusion matrix.
        self.confusion_mat = confusion_matrix(actual_result, pred_result)
        # initialize each criteria.
        self.accuracy = 0
        self.recall = {}
        self.f_score = {}
        self.precision = {}

    def heatMap(self):
        plt.title('Confusion Matrix')
        sns.heatmap(self.confusion_mat, annot = True, fmt = 'g')
        plt.show()
        
    def Accuracy(self):
        # used to calculate accuracy of whole model.
        acc = accuracy_score(self.actual_result, self.pred_result)
        return acc

    def Recall(self):
        # used to calculate micro, macro and weighted recall for model.
        # add result into dictionary.
        self.recall['Micro_Recall'] = recall_score(self.actual_result, self.pred_result, average='micro')
        self.recall['Macro_Recall'] = recall_score(self.actual_result, self.pred_result, average='macro')
        self.recall['Weighted_Recall'] = recall_score(self.actual_result, self.pred_result, average='weighted')
        return self.recall

    def F_score(self):
        # used to calculate micro, macro and weighted f-score for model.
        # add result into dictionary.
        self.f_score['Micro_F_Score'] = f1_score(self.actual_result, self.pred_result, average='micro')
        self.f_score['Macro_F_Score'] = f1_score(self.actual_result, self.pred_result, average='macro')
        self.f_score['Weighted_F_Score'] = f1_score(self.actual_result, self.pred_result, average='weighted')
        return self.f_score

    def Precision(self):
        # used to calculate micro, macro and weighted precision for model.
        # add result into dictionary.
        self.precision['Micro_Precision'] = precision_score(self.actual_result, self.pred_result, average='micro')
        self.precision['Macro_Precision'] = precision_score(self.actual_result, self.pred_result, average='macro')
        self.precision['Weighted_Precision'] = precision_score(self.actual_result, self.pred_result, average='weighted')
        return self.precision

    def calculate_all(self):
        # run all code and print result.
        print('-------Confusion Matrix-------')
        print(self.confusion_mat)
        self.heatMap()
        print()
        print('----------Accuracy----------')
        self.accuracy = self.Accuracy()
        print(self.accuracy)
        print()
        print('-----------Recall-----------')
        self.recall = self.Recall()
        for i in self.recall.items():
            print(str(i[0]) + ': ' + str(i[1]))
            print()
        print('-----------F-Score-----------')
        self.f_score = self.F_score()
        for i in self.f_score.items():
            print(str(i[0]) + ': ' + str(i[1]))
            print()
        print('----------Precision----------')
        self.precision = self.Precision()
        for i in self.precision.items():
            print(str(i[0]) + ': ' + str(i[1]))
            print()


