import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Score import score  # import the score function from Score.py

train_set = pd.read_csv('training.csv')
train, vad = train_test_split(train_set, test_size=0.25, random_state=1)
x_vad = vad.iloc[:, :-1]
y_vad = list(vad.iloc[:, -1].values)


def bayes(train, test):
    # Calculate the prior probability
    label = train.columns.tolist()[-1]
    target = train[label].value_counts()
    PriorProb = {}
    for i in target.index:
        PriorProb[i] = target[i] / sum(target)

    # Calculate the condition probability
    features = []
    values = []
    for w in train.columns.tolist()[:-1]:
        features.extend(len(train[w].unique()) * [w])
        values.extend(train[w].unique().tolist())
    ConditionProb = pd.DataFrame(np.zeros((len(values), 2)), columns=train[label].value_counts().index.tolist(),
                                 index=[features, values])
    for i in target.index:
        sub_df1 = train[train.iloc[:, -1] == i]
        for j in train.columns.tolist()[:-1]:
            sub_df2 = sub_df1[j].value_counts()
            for k in train[j].value_counts().index.tolist():
                if k not in sub_df2.index.tolist():
                    sub_df2[k] = 0
                ConditionProb.loc[(j, k)].loc[i] = (sub_df2[k] + 1) / (sum(sub_df2) + len(sub_df2.index.tolist()))

    # Predict the result by calculating the probability
    total_prob = {}
    pred_result = []
    label = train.columns.tolist()[-1]
    target = train[label].value_counts()
    for k in range(len(test)):
        for i in target.index:
            p1, p2 = 1, 1
            for j in train.columns.tolist()[:-1]:
                p1 *= ConditionProb[i][j][test.iloc[k, :][j]]
            p2 = p1 * PriorProb[i]
            total_prob[i] = p2
        pred_result.append(max(total_prob, key=total_prob.get))
    return pred_result


# Fit the Bayes model and print the model score
print('Bayes Model')
pred_result = bayes(train, x_vad)
score(pred_result, y_vad)
