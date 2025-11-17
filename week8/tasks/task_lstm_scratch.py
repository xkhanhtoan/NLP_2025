# task_lstm_scratch.py
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/hwu"
MAX_VOCAB = 20000
MAXLEN = 200
EMB_DIM = 128

def clean(t):
    t = t.lower()
    t = re.sub(r"[^\w\s]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

X_train = [clean(x) for x in train["text"].astype(str)]
X_test  = [clean(x) for x in test["text"].astype(str)]
y_train = train["category"].astype(str).tolist()
y_test  = test["category"].astype(str).tolist()

tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tok.fit_on_texts(X_train)

Xtr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=MAXLEN)
Xte = pad_sequences(tok.texts_to_sequences(X_test), maxlen=MAXLEN)

labels = sorted(set(y_train))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

ytr = np.array([label2id[l] for l in y_train])
yte = np.array([label2id[l] for l in y_test])

model = Sequential([
    Embedding(MAX_VOCAB, EMB_DIM, input_length=MAXLEN),
    LSTM(128),
    Dropout(0.5),
    Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

es = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(Xtr, ytr, epochs=10, batch_size=32,
          validation_split=0.1, callbacks=[es], verbose=2)

pred = model.predict(Xte)
yp = np.argmax(pred, axis=1)

print("\n===== REPORT =====")
print(classification_report(yte, yp, target_names=labels))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(yte, yp))
