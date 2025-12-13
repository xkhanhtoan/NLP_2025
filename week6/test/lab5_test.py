# test_text_classification.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from src.models.text_classification import TextClassification
from src.models.my_vectorizer import MyVectorizer

# 1. Dataset
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,       # 20% test
    random_state=42,
    shuffle=True,
    stratify=labels
)

# 3. Initialize vectorizer and classifier
vectorizer = MyVectorizer()
tcf = TextClassification(vectorizer=vectorizer)

# 4. Train
tcf.fit(X_train, y_train)

# 5. Predict
y_pred = tcf.predict(X_test)

# 6. Evaluate
metrics = tcf.evaluate(y_test, y_pred)

# 7. Print results
print("X_test:", X_test)
print("y_test:", y_test)
print("y_pred:", y_pred)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
