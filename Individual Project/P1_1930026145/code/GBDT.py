import pandas as pd
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from Score import score # import the score function from Score.py

# Make the one-hot encoder
train_set = pd.read_csv('training.csv')
train, vad = train_test_split(train_set, test_size=0.25,random_state=12)
x_train_ohe = pd.get_dummies(train.iloc[:, :-1])
y_train_ohe = train.iloc[:, -1]
x_vad_ohe = pd.get_dummies(vad.iloc[:, :-1])
y_vad = list(vad.iloc[:, -1].values)

# Do the grid research
model = GradientBoostingClassifier(n_estimators=50, max_features=5)
parameters = {'n_estimators': [10, 200, 10], 'max_features': [1, 6, 1]}
gbdt = GridSearchCV(model, parameters, cv=5)
gbdt.fit(x_train_ohe, y_train_ohe)
print('Best parameters:', gbdt.best_params_)

# Use the best hyperparameters to fit the GBDT model
model = GradientBoostingClassifier(n_estimators=200, max_features=6)
GBDT = model.fit(x_train_ohe,y_train_ohe)
pred_result = list(GBDT.predict(x_vad_ohe))
print('Gradient Boosting Decision Tree')
score(pred_result,y_vad)