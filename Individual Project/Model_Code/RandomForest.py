import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Make the label encoder
data = pd.read_csv('Dataset/training.csv')

class RandomForest():
    def __init__(self, data):
        self.data = data
    
    def spilt_data(self):
        train, valid = train_test_split(self.data, test_size = 0.3, random_state = 22)
        return train, valid

    def label_encoder(self, train, valid):
        # Make the label encoder
        labEcd = LabelEncoder()
        # initialize x_train_encoder, x_valid_encoder.
        x_train_encoder, x_valid_encoder = pd.DataFrame(), pd.DataFrame()
        for i in train.columns[ :-1]:
            x_train_encoder[i] = labEcd.fit_transform(train[i])
            x_valid_encoder[i] = labEcd.fit_transform(valid[i])

        y_train_encoder = train.iloc[:, -1]
        y_valid_list = list(valid.iloc[:, -1]. values)
        return x_train_encoder, y_train_encoder, x_valid_encoder, y_valid_list

    def get_valid_y(self):
        return self.label_encoder(self.spilt_data()[0], self.spilt_data()[1])[3]

    def grid_search(self, x_train_encoder, y_train_encoder):
        model = RandomForestClassifier(n_estimators = 50, max_features = 5)
        parameters = {'n_estimators': [10, 200, 10], 'max_features': [1, 6, 1]}
        gs = GridSearchCV(model, parameters, cv=5)
        gs.fit(x_train_encoder, y_train_encoder)
        print('Best parameters:', gs.best_params_)
        return gs.best_params_

    def random_forest_fit(self, best_para, x_train_encoder, y_train_encoder):
        # Use the best hyperparameters to fit the Random Forest model
        model = RandomForestClassifier(n_estimators = best_para['n_estimators'], max_features = best_para['max_features'])
        model.fit(x_train_encoder, y_train_encoder)
        return model

    def predict(self, model, x_valid_encoder):
        result = model.predict(x_valid_encoder)
        return result

    def get_result(self):
        train, valid = self.spilt_data()
        x_t_e, y_t_e, x_v_e, _ = self.label_encoder(train, valid)
        best_para = self.grid_search(x_t_e, y_t_e)
        model = self.random_forest_fit(best_para, x_t_e, y_t_e)
        result = self.predict(model, x_v_e)
        print('--------Random_Forest Model--------')
        print(result)
        print('Done')
        return result

