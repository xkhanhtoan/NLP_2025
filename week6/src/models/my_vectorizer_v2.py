from typing import List
import numpy as np
import math
import re

STOPWORDS = {
    "is","am","are","the","a","an","and","or","to","of","in","it","this","that","so"
}

class MyVectorizerV2:
    def __init__(self, min_df: int = 1):
        self.vocab = {}
        self.idf = {}
        self.min_df = min_df
        self.fitted = False

    def _preprocess(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        words = [w for w in text.split() if w not in STOPWORDS]
        return words

    def fit(self, texts: List[str]):
        N = len(texts)
        df = {}

        for text in texts:
            words = set(self._preprocess(text))
            for w in words:
                df[w] = df.get(w, 0) + 1

        df = {w:c for w,c in df.items() if c >= self.min_df}
        self.vocab = {w:i for i,w in enumerate(sorted(df))}
        self.idf = {w: math.log((N+1)/(df[w]+1)) + 1 for w in self.vocab}

        self.fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            words = self._preprocess(text)
            tf = {}
            for w in words:
                if w in self.vocab:
                    tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                X[i, self.vocab[w]] = (c/len(words)) * self.idf[w]
        return X

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)