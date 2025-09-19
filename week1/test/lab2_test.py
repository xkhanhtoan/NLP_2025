from lab1.src.preprocessing.regex_tokenizer import RegexTokenizer
from lab2.src.representations.count_vectorizer import CountVectorizer

def test_count_vectorizer():
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    X = vectorizer.fit_transform(corpus)

    print("Vocabulary:", vectorizer.vocabulary_)
    print("\nDocument-term matrix:")
    for row in X:
        print(row)

if __name__ == "__main__":
    test_count_vectorizer()
