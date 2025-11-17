# task_tfidf_lr.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/hwu"

# Load data
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

X_train = train["text"].astype(str)
y_train = train["category"].astype(str)

X_test = test["text"].astype(str)
y_test = test["category"].astype(str)

# Model pipeline
model = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000)
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# =======================
#  LOG PHÂN TÍCH KẾT QUẢ
# =======================

print("\n===== CLASSIFICATION REPORT =====\n")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====\n")
print(confusion_matrix(y_test, y_pred))

print("\n===== MISCLASSIFIED EXAMPLES (max 10) =====\n")
mistakes = (y_pred != y_test)

wrong_idx = test[mistakes].index.tolist()

for i in wrong_idx[:10]:
    print("------")
    print(f"Text      : {test.loc[i, 'text']}")
    print(f"True      : {y_test[i]}")
    print(f"Predicted : {y_pred[i]}")
    print()
