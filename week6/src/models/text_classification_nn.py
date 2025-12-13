from sklearn.neural_network import MLPClassifier
from typing import List

class TextClassificationNN:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.model = MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=500,
            random_state=42
        )

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts: List[str]):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)