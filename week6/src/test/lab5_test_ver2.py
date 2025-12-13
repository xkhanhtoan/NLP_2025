# test/lab5_task4_test.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.models.my_vectorizer_v2 import MyVectorizerV2
from src.models.text_classification import TextClassification
from src.models.text_classification_nb import TextClassificationNB
from src.models.text_classification_nn import TextClassificationNN

texts = [
    "This movie is fantastic and I love it",
    "I hate this film it is terrible",
    "The acting was superb and great",
    "Absolutely boring waste of time",
    "Highly recommend this masterpiece",
    "So bad could not finish watching"
]
labels = [1,0,1,0,1,0]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42, stratify=labels
)

# 1. Improved TF-IDF + Logistic Regression
vec1 = MyVectorizerV2(min_df=1)
lr = TextClassification(vec1)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# 2. Naive Bayes
vec2 = MyVectorizerV2(min_df=1)
nb = TextClassificationNB(vec2)
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

# 3. Neural Network
vec3 = MyVectorizerV2(min_df=1)
nn = TextClassificationNN(vec3)
nn.fit(X_train, y_train)
pred_nn = nn.predict(X_test)

print("Improved LR accuracy:", accuracy_score(y_test, pred_lr))
print("Naive Bayes accuracy:", accuracy_score(y_test, pred_nb))
print("Neural Network accuracy:", accuracy_score(y_test, pred_nn))
