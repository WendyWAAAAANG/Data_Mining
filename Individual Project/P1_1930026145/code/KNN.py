import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Score import score  # import the score function from Score.py

# Data preprocessing
# Read the dataset and make the one-hot encoder
# Divide it to the train and validate to fit and validate the model
# Ratio of training set to vad set is 0.75:0.25
train_set = pd.read_csv('training.csv')
target_name = train_set.columns[-1]
train_target = train_set[target_name]
train_dummies = pd.get_dummies(train_set.iloc[:, :-1])
train_set = pd.concat([train_dummies, train_target], axis=1)
train, vad = train_test_split(train_set, test_size=0.25, random_state=70)
x_vad = vad.iloc[:,:-1]
y_vad = list(vad.iloc[:, -1].values)


def kNN(train, test, k):
    target_name = train.columns.tolist()[-1]
    labels = train.iloc[:, -1]
    pred_result = []
    for i in range(len(test)):
        # Calculate the distance between the test point and every train point
        diff = train.iloc[:, :-1].add(-1 * test.iloc[i, :], axis=1)
        distance = pd.DataFrame(np.sqrt(np.sum(np.square(diff), axis=1)), columns=['distance'])
        # Sort the distance from close to far and count the target from k closest points
        # The target of test point is decided the target of majority k closet ponits
        df = pd.concat([distance, labels], axis=1)
        sorted_df = df.sort_values(by='distance', ascending=True)
        series = sorted_df.iloc[:k, :][target_name].value_counts()
        pred_result.append(series.sort_values(ascending=False).index[0])
    return pred_result


# Fit the KNN model and print the model score
print('KNN Model')
pred_result = knn(train, x_vad, 6)
score(pred_result, y_vad)
