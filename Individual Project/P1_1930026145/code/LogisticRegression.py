import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Score import score # import the score function from Score.py

# Make the one-hot encoder
train_set = pd.read_csv('training.csv')
train, vad = train_test_split(train_set, test_size=0.25, random_state=1)
x_train_ohe = pd.get_dummies(train.iloc[:, :-1])
y_train_ohe = train.iloc[:, -1]
x_vad_ohe = pd.get_dummies(vad.iloc[:, :-1])
y_vad = list(vad.iloc[:, -1].values)

model = LogisticRegression()
lgr = model.fit(x_train_ohe,y_train_ohe)
pred_result = list(lgr.predict(x_vad_ohe))
score(pred_result,y_vad)