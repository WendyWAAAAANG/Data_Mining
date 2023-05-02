import pandas as pd
from treelib import Tree
from sklearn.model_selection import train_test_split
from Score import score  # import the score function from Score.py

# Data preprocessing
# Read the dataset and divide it to the train and validate to fit and validate the model
# Ratio of training set to validate set is 0.75:0.25
train_set = pd.read_csv('training.csv')
train, vad = train_test_split(train_set, test_size=0.25, random_state=17)
x_vad = vad.iloc[:, :-1]
y_vad = list(vad.iloc[:, -1].values)


# Calculate the Gini coefficient for vector x
def Gini(x):
    sum = 0
    for i in x:
        sum += i ** 2
    return 1 - sum


# Calculate the Gini coefficient of features and target
def Feature_Labels_Gini(data):
    # Get the Gini coefficient of target
    label_name = data.columns[-1]
    labels = data[label_name].value_counts()
    prob_label = [labels[0] / sum(labels), labels[1] / sum(labels)]
    Gini_labels = Gini(prob_label)

    # Get the Gini coefficient of features
    Gini_feature = {}
    for i in data.columns[:-1]:
        sub_df1 = data[i].value_counts()
        gini = 0
        for j in sub_df1.index:
            prob_feature_labels = sub_df1[j] / sum(sub_df1)
            sub_df2 = data.iloc[:, -1][data[i] == j].value_counts()
            if len(sub_df2) < 2:
                gini += 0
            else:
                prob_feature = [sub_df2[0] / sum(sub_df2), sub_df2[1] / sum(sub_df2)]
                gini += prob_feature_labels * Gini(prob_feature)
        Gini_feature[i] = gini
    return Gini_labels, Gini_feature


# Select the feature to construct the decision tree in this turn
def Feature_Decision(Gini_feature, Gini_labels):
    diff_gini_feature = {}
    for i in Gini_feature.keys():
        diff_gini_feature[i] = Gini_labels - Gini_feature[i]
    decide_feature = max(diff_gini_feature, key=diff_gini_feature.get)
    return decide_feature


# Construct the decision tree, we store our tree by recording the identifier of every node and its parent
# For the leave node, we also need to store the target result --- 'acc' or 'unacc'
def Construct_Tree(data, parent, tree, feature_order):
    # If there is only one kind of target in the data, we stop constructing
    if len(data.iloc[:, -1].unique()) == 1:
        tree.create_node(tag=data.iloc[:, -1].iloc[0], parent=parent, identifier=parent + '_result',
                         data=data.iloc[:, -1].iloc[0])
        return
    # If there is only one feature in the data, we also stop constructing
    elif len(data.columns) == 2:
        for i in data.iloc[:, 0].value_counts().index:
            tree.create_node(tag=data.columns[0] + '_' + i, parent=parent,
                             identifier=parent + data.columns[0] + '_' + i)
            dic = dict(data[data.iloc[:, 0] == i].iloc[:, -1].value_counts())
            tree.create_node(tag=max(dic, key=dic.get), parent=parent + data.columns[0] + '_' + i,
                             identifier=parent + data.columns[0] + '_' + i + '_result',
                             data=max(dic, key=dic.get))
        return
    gini_labels, gini_feature = Feature_Labels_Gini(data)
    decided_feature = Feature_Decision(gini_feature, gini_labels)
    # If the Gini coefficient of selected feature in the turn is larger than 0.4, we stops constructing (pruning)
    if gini_feature[decided_feature] >= 0.4:
        dic1 = dict(data.iloc[:-1].value_counts())
        tree.create_node(tag=max(dic1, key=dic1.get), parent=parent,
                         identifier=parent + '_result', data=max(dic1, key=dic1.get))
        return
    for k in data[decided_feature].value_counts().index:
        if decided_feature not in feature_order:
            feature_order.append(decided_feature)
        new_data = data[data[decided_feature] == k].drop(columns=decided_feature, axis=1)
        tree.create_node(tag=decided_feature + '_' + k, identifier=parent + '_' + decided_feature + '_' + k,
                         parent=parent)
        # We use recursion to construct the decision tree
        Construct_Tree(new_data, parent + '_' + decided_feature + '_' + k, tree, feature_order)


# Use the decision tree we constructed before to predict the result for validation data
def predict(test, tree, feature_order):
    pred_list = []
    for i in range(len(test)):
        item = 'root'
        for j in feature_order:
            item = item + '_' + j + '_' + test[j].iloc[i]
            if tree.get_node(item) == None:
                pred_list.append('acc')
                break
            elif tree.get_node(item + '_result') == None:
                continue
            else:
                pred_list.append(tree.get_node(item + '_result').data)
            break
    return pred_list


def decisiontree(train, test):
    tree = Tree()
    tree.create_node(tag='root', identifier='root', data=0)
    feature_order = []
    Construct_Tree(train, 'root', tree, feature_order)
    pred_result = predict(test, tree, feature_order)
    return pred_result


# Fit the Decision Tree model and print the model score
print('Decision Tree Model')
pred_result = decisiontree(train, x_vad)
score(pred_result, y_vad)
