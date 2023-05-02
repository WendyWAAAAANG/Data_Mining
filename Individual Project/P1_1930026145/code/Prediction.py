import pandas as pd
from sklearn.svm import SVC

# Use SVM model to predict the target for the test set
train_set = pd.read_csv('training.csv')
test_set = pd.read_csv('test.csv')
x_train = pd.get_dummies(train_set.iloc[:,:-1])
y_train = train_set.iloc[:,-1]
test_dummies = pd.get_dummies(test_set)
model = SVC(kernel='poly',C=1)
svm = model.fit(x_train,y_train)
pred_result = svm.predict(test_dummies)
test_set['evaluation'] = pred_result
test_set.to_csv("test<SVM>.csv", index=False, sep=',')