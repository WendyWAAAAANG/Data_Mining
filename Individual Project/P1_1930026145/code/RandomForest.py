import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
from Score import score # import the score function from Score.py

# Make the label encoder
train_set = pd.read_csv('training.csv')
train, vad = train_test_split(train_set, test_size=0.25, random_state=1)
# Make the label encoder
le = LabelEncoder()
x_train_le, x_vad_le = pd.DataFrame(), pd.DataFrame()
for i in train.columns[:-1]:
    x_train_le[i] = le.fit_transform(train[i])
    x_vad_le[i] = le.fit_transform(vad[i])
y_train_le = train.iloc[:, -1]
y_vad = list(vad.iloc[:, -1].values)

# Do the grid research
model = RandomForestClassifier(n_estimators=50, max_features=5)
parameters = {'n_estimators': [10, 200, 10], 'max_features': [1, 6, 1]}
rf = GridSearchCV(model, parameters, cv=5)
rf.fit(x_train_le, y_train_le)
print('Best parameters:', rf.best_params_)

# Use the best hyperparameters to fit the Random Forest model
model = RandomForestClassifier(n_estimators=200, max_features=6)
model.fit(x_train_le, y_train_le)
pred_result = list(model.predict(x_vad_le))
print('RandomForest')
score(pred_result, y_vad)


