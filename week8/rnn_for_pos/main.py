import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# ======================
# 1. LOAD CONLLU
# ======================
def load_conllu(path):
    sentences = []
    sent = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if sent:
                    sentences.append(sent)
                    sent = []
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue

            word = cols[1]
            tag = cols[3]
            sent.append((word, tag))

    if sent:
        sentences.append(sent)

    return sentences


# ======================
# 2. BUILD VOCAB
# ======================
def build_vocab(sentences):
    word_counter = Counter()
    tag_counter = Counter()

    for sent in sentences:
        for w, t in sent:
            word_counter[w.lower()] += 1
            tag_counter[t] += 1

    word_to_ix = {"<PAD>": 0, "<UNK>": 1}
    for w in word_counter:
        word_to_ix[w] = len(word_to_ix)

    tag_to_ix = {"<PAD>": 0}
    for t in tag_counter:
        tag_to_ix[t] = len(tag_to_ix)

    return word_to_ix, tag_to_ix


# ======================
# 3. DATASET
# ======================
class POSDataset(Dataset):
    def __init__(self, data, word_to_ix, tag_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        words = [self.word_to_ix.get(w.lower(), 1) for w, _ in sent]
        tags = [self.tag_to_ix[t] for _, t in sent]
        return torch.tensor(words), torch.tensor(tags)


def collate_fn(batch):
    sents, tags = zip(*batch)
    sents = pad_sequence(sents, batch_first=True, padding_value=0)
    tags = pad_sequence(tags, batch_first=True, padding_value=0)
    return sents, tags


# ======================
# 4. MODEL
# ======================
class RNNTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)


# ======================
# 5. TRAIN & EVAL
# ======================
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(-1)

            mask = y != 0
            correct += (pred[mask] == y[mask]).sum().item()
            total += mask.sum().item()

    if total == 0:
        return 0.0

    return correct / total


def train(model, train_loader, dev_loader, device):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(5):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            out = model(x)
            loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
            loss.backward()
            opt.step()

            total_loss += loss.item()

        acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Dev Acc: {acc:.4f}")


# ======================
# 6. MAIN
# ======================
def main():
    TRAIN_PATH = r"C:\Users\Z\Desktop\2025-code\NLP\week9\rnn_for_pos\data\en_ewt-ud-train.conllu"

    print("Loading data...")
    train_data = load_conllu(TRAIN_PATH)

    # ✅ dùng 1000 câu train làm dev (ổn định, đủ demo)
    dev_data = train_data[:1000]
    train_data = train_data[1000:]

    word_to_ix, tag_to_ix = build_vocab(train_data)

    print("Vocab size:", len(word_to_ix))
    print("Tag size:", len(tag_to_ix) - 1)

    train_ds = POSDataset(train_data, word_to_ix, tag_to_ix)
    dev_ds = POSDataset(dev_data, word_to_ix, tag_to_ix)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RNNTagger(
        vocab_size=len(word_to_ix),
        emb_dim=100,
        hidden_dim=128,
        out_dim=len(tag_to_ix)
    ).to(device)

    train(model, train_loader, dev_loader, device)


if __name__ == "__main__":
    main()
