from river import forest, tree

class Classifier:
    def __init__(self, n_models=10):
        self.n_models=n_models
        #self.model= forest.ARFClassifier(n_models=self.n_models, seed=42)
        self.model= tree.HoeffdingTreeClassifier()
        
    def predict(self, x):
        x_dict = {f"f{i}": float(v) for i, v in enumerate(x)}
        y_pred = self.model.predict_one(x_dict)
        return y_pred

    def learn(self, x, y):
        x_dict = {f"f{i}": float(v) for i, v in enumerate(x)}
        self.model.learn_one(x_dict, y)

    def reset(self):
        self.model = self._create_model()

    def predict_proba(self, x):
        x_dict = {f"f{i}": float(v) for i, v in enumerate(x)}
        proba_dict = self.model.predict_proba_one(x_dict) #{class_id: probability}
        return proba_dict


