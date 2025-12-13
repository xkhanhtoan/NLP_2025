from src.models.my_vectorizer import MyVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List

class TextClassification:
    def __init__(self, vectorizer: MyVectorizer):
        self.vectorizer = vectorizer
        self._model = LogisticRegression(solver="liblinear")
    
    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self._model.fit(X,labels)

    def predict(self, texts: List[str]):
        X = self.vectorizer.transform(texts)   
        return self._model.predict(X)
    
    def evaluate(self, y_true: List[int], y_pred: List[int]):
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))
        }
        return metrics


    




        