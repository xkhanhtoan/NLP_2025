# task_w2v_dense.py
import os
import re
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/hwu"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r"[^\w\s]"," ",text)
    return re.sub(r"\s+"," ", text).strip()

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

X_train_raw = train['text'].astype(str).tolist()
X_test_raw  = test['text'].astype(str).tolist()
y_train = train['category'].astype(str).tolist()
y_test  = test['category'].astype(str).tolist()

X_train = [clean(t) for t in X_train_raw]
X_test  = [clean(t) for t in X_test_raw]

sentences = [s.split() for s in X_train]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, seed=SEED, workers=1, epochs=10)

def avg_vec(text):
    toks = text.split()
    vecs = [w2v.wv[w] for w in toks if w in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=float)
    return np.mean(vecs, axis=0)

Xtr = np.vstack([avg_vec(t) for t in X_train])
Xte = np.vstack([avg_vec(t) for t in X_test])

# label encode simple mapping
labels = sorted(set(y_train))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
ytr = np.array([label2id[l] for l in y_train])
yte = np.array([label2id[l] for l in y_test])

model = Sequential([Dense(128, activation="relu", input_shape=(w2v.vector_size,)), Dropout(0.5), Dense(len(labels), activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(Xtr, ytr, epochs=20, batch_size=32, validation_split=0.1, callbacks=[es], verbose=2)
y_pred_probs = model.predict(Xte)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n===== CLASSIFICATION REPORT =====\n")
print(classification_report(yte, y_pred, target_names=labels))
print("\n===== CONFUSION MATRIX =====\n")
print(confusion_matrix(yte, y_pred))
print("\n===== MISCLASSIFIED EXAMPLES (max 10) =====\n")
mistakes = np.where(yte != y_pred)[0]
for i in mistakes[:10]:
    print("------")
    print(f"Text      : {X_test_raw[i]}")
    print(f"True      : {id2label[yte[i]]}")
    print(f"Predicted : {id2label[y_pred[i]]}\n")
# optionally save models
w2v.save(os.path.join(MODEL_DIR, "w2v.model"))
model.save(os.path.join(MODEL_DIR, "w2v_dense.h5"))
