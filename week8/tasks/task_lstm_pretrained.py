# task_lstm_pretrained.py
import os
import re
import sys
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------
# Config / Paths
# ------------------------
DATA_DIR = "data/hwu"
W2V_PATH  = "models/w2v.model"     # <-- path to your pretrained .model (gensim Word2Vec)
MAXLEN    = 200
MAX_VOCAB = 20000                  # define max vocab used by Tokenizer
MODEL_OUT = "models/lstm_pretrained.keras"

# ------------------------
# Quick checks
# ------------------------
if not os.path.exists(W2V_PATH):
    print(f"[ERROR] Missing {W2V_PATH}. Train Word2Vec in task 2 first or update W2V_PATH.")
    raise SystemExit

train_csv = os.path.join(DATA_DIR, "train.csv")
test_csv  = os.path.join(DATA_DIR, "test.csv")
if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    print(f"[ERROR] train.csv or test.csv not found in {DATA_DIR}.")
    raise SystemExit

# ------------------------
# Reproducibility / seeds
# ------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------
# Utilities
# ------------------------
def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---------------------------
# 1) Load data
# ---------------------------
train = pd.read_csv(train_csv)
test  = pd.read_csv(test_csv)

# Expect 'text' and 'category' columns
if 'text' not in train.columns or 'category' not in train.columns:
    print("[ERROR] train.csv must contain 'text' and 'category' columns.")
    raise SystemExit
if 'text' not in test.columns or 'category' not in test.columns:
    print("[ERROR] test.csv must contain 'text' and 'category' columns.")
    raise SystemExit

X_train_raw = train['text'].astype(str).tolist()
X_test_raw  = test['text'].astype(str).tolist()
y_train_raw = train['category'].astype(str).tolist()
y_test_raw  = test['category'].astype(str).tolist()

X_train = [clean(t) for t in X_train_raw]
X_test  = [clean(t) for t in X_test_raw]
y_train = [str(l) for l in y_train_raw]
y_test  = [str(l) for l in y_test_raw]

print(f"Loaded data: {len(X_train)} train examples, {len(X_test)} test examples.")

# ---------------------------
# 2) Tokenizer + sequences
# ---------------------------
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = min(MAX_VOCAB, len(word_index) + 1)  # +1 because index starts at 1

Xtr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAXLEN, padding="post")
Xte = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAXLEN, padding="post")

print(f"Tokenizer: word_index size = {len(word_index)}, using vocab_size = {vocab_size}")

# ---------------------------
# 3) Load Word2Vec pretrained (.model)
# ---------------------------
print(f"Loading Word2Vec model from {W2V_PATH} ...")
w2v = Word2Vec.load(W2V_PATH)
EMB_DIM = w2v.wv.vector_size
print(f"Loaded Word2Vec. Vector size = {EMB_DIM}")

# ---------------------------
# 4) Build embedding matrix
# ---------------------------
embedding_matrix = np.zeros((vocab_size, EMB_DIM), dtype=np.float32)

hits = 0
misses = 0
not_found_samples = []
for word, idx in word_index.items():
    if idx >= vocab_size:
        continue
    if word in w2v.wv.key_to_index:
        embedding_matrix[idx] = w2v.wv[word]
        hits += 1
    else:
        misses += 1
        if len(not_found_samples) < 50:
            not_found_samples.append(word)

print(f"Embedding matrix shape = {embedding_matrix.shape} | hits = {hits} | misses = {misses}")
if hits + misses > 0:
    print(f"Coverage: {hits / (hits + misses):.3f}")
print("Sample tokens not found in w2v:", not_found_samples[:20])

# If coverage is low, training embedding (trainable=True) is helpful.

# ---------------------------
# 5) Encode labels (robust handling of unseen test labels)
# ---------------------------
labels = sorted(set(y_train))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

ytr = np.array([label2id[l] for l in y_train], dtype=np.int32)

# Map test labels; exclude unseen ones
yte_mapped = []
valid_test_idx = []
for i, lab in enumerate(y_test):
    if lab in label2id:
        yte_mapped.append(label2id[lab])
        valid_test_idx.append(i)
    else:
        print(f"[WARN] test idx {i} label '{lab}' not in train labels; excluding from evaluation")

if len(yte_mapped) == 0:
    print("[ERROR] After excluding unseen labels, no test examples remain. Aborting.")
    raise SystemExit

yte = np.array(yte_mapped, dtype=np.int32)
Xte_eval = Xte[valid_test_idx] if len(valid_test_idx) < len(Xte) else Xte
X_test_raw_eval = [r for i, r in enumerate(X_test_raw) if i in valid_test_idx] if len(valid_test_idx) < len(X_test_raw) else X_test_raw

print(f"Evaluation: {Xte_eval.shape[0]} examples after excluding unseen labels.")

# ---------------------------
# 6) Build model (embedding trainable, mask_zero)
# ---------------------------
num_classes = len(labels)
model = Sequential([
    Embedding(vocab_size, EMB_DIM, weights=[embedding_matrix],
              input_length=MAXLEN, trainable=True, mask_zero=True),

    LSTM(128),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

es = EarlyStopping(patience=3, restore_best_weights=True, verbose=1)

# ---------------------------
# 7) Train
# ---------------------------
print("Training model ...")
model.fit(Xtr, ytr,
          epochs=12,
          batch_size=32,
          validation_split=0.1,
          callbacks=[es],
          verbose=2,
          shuffle=True)

# ---------------------------
# 8) Evaluate
# ---------------------------
y_pred_probs = model.predict(Xte_eval)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n===== CLASSIFICATION REPORT =====\n")
print(classification_report(yte, y_pred, target_names=labels, zero_division=0))

print("\n===== CONFUSION MATRIX =====\n")
print(confusion_matrix(yte, y_pred))

print("\n===== MISCLASSIFIED EXAMPLES (max 10) =====\n")
mistakes = np.where(yte != y_pred)[0]
for idx in mistakes[:10]:
    i = idx
    print("------")
    print(f"Text      : {X_test_raw_eval[i]}")
    print(f"True      : {id2label[yte[i]]}")
    print(f"Predicted : {id2label[y_pred[i]]}\n")

# ---------------------------
# 9) Save model
# ---------------------------
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
try:
    model.save(MODEL_OUT)
    print(f"Saved LSTM model to {MODEL_OUT}")
except Exception as e:
    print("Failed to save model:", e)
