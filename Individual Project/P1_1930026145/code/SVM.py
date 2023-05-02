import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from Score import score # import the score function from Score.py

train_set = pd.read_csv('training.csv')
train_dummies = pd.get_dummies(train_set.iloc[:, :-1])
train_target = train_set.iloc[:, -1]


# k fold validation for SVM Model
kf = KFold(n_splits=5)
for k, (train_index, vad_index) in enumerate(kf.split(train_set)):
    x_train, x_vad, y_train, y_vad = train_dummies.values[train_index], train_dummies.values[vad_index], train_target[
        train_index], list(train_target[vad_index])
    model = SVC(kernel='poly', C=1)
    svm = model.fit(x_train, y_train)
    y_pred = list(svm.predict(x_vad))
    
    print('SVM {} fold'.format(k + 1))
    score(y_pred,y_vad)