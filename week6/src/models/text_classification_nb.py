from sklearn.naive_bayes import MultinomialNB
from typing import List

class TextClassificationNB:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.model = MultinomialNB()

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts: List[str]):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)