from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce

class GBDT():
    def __init__(self, x_train, x_valid, y_train):
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train

    def encode_data(self):
        # Encode training features with ordinal encoding 
        encoder = ce.OrdinalEncoder(cols = self.x_train)
        self.x_train = encoder.fit_transform(self.x_train)
        self.x_valid = encoder.transform(self.x_valid)

    def fit_model(self):
        clf_GBDT = GradientBoostingClassifier(random_state = 22, n_estimators = 80, max_features = 5)
        clf_GBDT = clf_GBDT.fit(self.x_train, self.y_train)
        return clf_GBDT
    
    def prediction(self, clf_GBDT):
        res_GBDT = clf_GBDT.predict(self.x_valid)
        return res_GBDT
    
    def get_result(self):
        print('-----------GBDT Model-----------')
        self.encode_data()
        clf = self.fit_model()
        res_GBDT = self.prediction(clf)
        print('Done')
        return res_GBDT