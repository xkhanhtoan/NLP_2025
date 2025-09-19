class Vectorizer:   
    def fit(self, corpus: list[str]) :
        pass
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)
