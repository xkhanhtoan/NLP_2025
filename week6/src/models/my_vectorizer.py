from typing import List
import numpy as np
import math

class MyVectorizer:

    def __init__(self):
        self.vocab: dict[str, int] = {} 
        self.idf: dict[str, float] = {}  
        self.fitted = False

    def fit(self, texts: List[str]):
        
        vocab_set = set()
        for text in texts:
            for word in text.lower().split():
                vocab_set.add(word)
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}

        N = len(texts)
        df = {word: 0 for word in self.vocab}
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                df[word] += 1

        self.idf = {word: math.log(N / (df[word])) for word in self.vocab}

        self.fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Vectorizer is not fitted. Call fit(...) first.")

        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            words = text.lower().split()
            tf = {}
            for w in words:
                if w in self.vocab:
                    tf[w] = tf.get(w, 0) + 1
            for w, count in tf.items():
                X[i, self.vocab[w]] = (count / len(words)) * self.idf[w]
        return X

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)