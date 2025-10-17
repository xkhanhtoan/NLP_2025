import gensim.downloader as api
from src.representations.regex_tokenizer import RegexTokenizer as Tokenizer
import numpy as np


class word_embedder:
    def __init__(self, model_name: str):

        self.model = api.load(model_name)
        self.tokenizer = Tokenizer()
        self.vector_size = self.model.vector_size

    def get_vector(self, word: str):
        return self.model[word] if word in self.model else None

    def get_similarity(self, word1: str, word2: str):
        if word1 in self.model and word2 in self.model:
            return self.model.similarity(word1, word2)
        else:
            return None

    def get_most_similar(self, word: str, top_n: int = 10):
        if word in self.model:
            return self.model.most_similar(word, topn=top_n)
        else:
            return None

    def embed_document(self, document: str):
        tokens = self.tokenizer.tokenize(document)
        vectors = []

        for token in tokens:
            vector = self.get_vector(token)
            if vector is not None:
                vectors.append(vector)

        if not vectors:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)
