import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Score import score

# Data preprocessing
# Read the dataset and divide it to the train and validate to fit and validate the model
# Ratio of training set to validate set is 0.75:0.25
# In the perceptron model, we use transform target 'acc' to 1 and 'unacc' to 0 in the train
train_set = pd.read_csv('training.csv')
target_name = train_set.columns[-1]
train, vad = train_test_split(train_set, test_size=0.25, random_state=40)
train[target_name][train[target_name] == 'acc'] = 1
train[target_name][train[target_name] == 'unacc'] = 0
train = pd.concat([pd.get_dummies(train.iloc[:, :-1]), train.iloc[:, -1]], axis=1)
x_vad = pd.get_dummies(vad.iloc[:, :-1])
y_vad = list(vad.iloc[:, -1].values)


def step(x):
    if x < 0:
        return 0
    else:
        return 1


def perceptron(train, test, iterations, learning_rate):
    features_num = len(train.columns[:-1])
    weight = -1 + 2 * np.random.rand(features_num, 1)  # Get the weight randomly at the beginning
    bias = 0
    labels = train.iloc[:, -1]
    iteration_number = 0
    for i in range(iterations):  # Do the iterations and update the weight after each iteration
        iteration_number += 1
        new_mat = np.zeros((len(train), 1))
        for j in range(len(train)):
            prediction = step(np.mat(train.iloc[j, :-1].values) * weight + bias)
            if labels.iloc[j] != prediction:
                bias = bias + learning_rate * labels.iloc[j]
                weight = weight + learning_rate * (labels.iloc[j] - prediction) * np.mat(train.iloc[j, :-1].values).T
            new_mat = np.mat(train.iloc[:, :-1]) * weight
            new_mat = np.where(new_mat < 0, 0, 1)
        # If all targets are true, we end it in advance and say it has converged
        if (new_mat.T[0] == labels.values).all():
            print('After {} iterations, it has converged!'.format(iteration_number))
            break
        print('{}/{} finished'.format(i + 1, iterations))

    # Predict the target in the validation set
    pred_result = []
    for k in range(len(test)):
        pred_result.append(step(np.mat(test.iloc[k, :].values) * weight + bias))
    pred_result = list(map(lambda x: 'unacc' if x == 0 else 'acc', pred_result))
    return pred_result


# Fit the Perceptron model and print the model score
print('Fitting the model takes some time,please keep patient')
pred_result = perceptron(train, x_vad, 20, 0.01)
print('Perceptron Model')
score(pred_result, y_vad)
