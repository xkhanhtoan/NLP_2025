from lab1.src.core.interfaces import Tokenizer
from lab2.src.core.interfaces import Vectorizer     
class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: dict[str, int] = {}
    def fit(self, corpus: list[str]):
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        sorted_tokens = sorted(unique_tokens)
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)}
    def transform(self, docs: list[str]):
        vectors = []
        vocab_size = len(self.vocabulary_)
        for doc in docs:
            vec = [0] * vocab_size
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    vec[idx] += 1
            vectors.append(vec)
        return vectors