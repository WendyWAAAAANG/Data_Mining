import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from Score import score  # import the score function from Score.py


# Ensemble model to improve the performance
def model_ensemble():
    result = pd.DataFrame()

    # SVM Model
    model = SVC(kernel='rbf', C=1)
    svm = model.fit(x_train_ohe, y_train_ohe)
    y_pred = svm.predict(x_vad_ohe)
    result['SVM'] = y_pred

    # Decision Tree Model
    model = DecisionTreeClassifier(criterion='entropy')
    dt = model.fit(x_train_le, y_train_le)
    y_pred = dt.predict(x_vad_le)
    result['DT'] = y_pred

    # Gradient Boosting Decision Tree Model
    model = GradientBoostingClassifier(n_estimators=150, max_features=6)
    gbdt = model.fit(x_train_le, y_train_le)
    y_pred = gbdt.predict(x_vad_le)
    result['GBDT'] = y_pred

    # Random Forest Model
    model = RandomForestClassifier(n_estimators=150, max_features=6)
    rf = model.fit(x_train_le, y_train_le)
    y_pred = rf.predict(x_vad_le)
    result['RF'] = y_pred

    # Bayes Model
    model = BernoulliNB()
    nb = model.fit(x_train_le, y_train_le)
    y_pred = nb.predict(x_vad_le)
    result['nb'] = y_pred

    # Ensemble all models and vote the target together
    pred_result = []
    for i in range(len(result)):
        ls = list(result.iloc[i, :])
        pred_result.append(max(ls, key=ls.count))
    return pred_result


train_set = pd.read_csv('training.csv')
train, vad = train_test_split(train_set, test_size=0.25, random_state=40)

# Make the label encoder
le = LabelEncoder()
x_train_le, x_vad_le = pd.DataFrame(), pd.DataFrame()
for i in train.columns[:-1]:
    x_train_le[i] = le.fit_transform(train[i])
    x_vad_le[i] = le.fit_transform(vad[i])
y_train_le = train.iloc[:, -1]

# Make the one-hot encoder
x_train_ohe = pd.get_dummies(train.iloc[:, :-1])
y_train_ohe = train.iloc[:, -1]
x_vad_ohe = pd.get_dummies(vad.iloc[:, :-1])

y_vad = list(vad.iloc[:, -1].values)

pred_result = model_ensemble()
print('Ensemble Model')
score(pred_result, y_vad)
