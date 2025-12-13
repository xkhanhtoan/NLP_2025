import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import defaultdict

# --- Cấu hình Siêu tham số ---
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
PAD_TAG_INDEX = -100 # Index đặc biệt cho padding nhãn, để CrossEntropyLoss bỏ qua

# --- 1. Lớp Dataset & Hàm Collate ---

class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences, self.tags = sentences, tags
        self.word_to_ix, self.tag_to_ix = word_to_ix, tag_to_ix
        self.UNK_INDEX = word_to_ix["<UNK>"]

    def __len__(self): return len(self.sentences)

    def __getitem__(self, index):
        sent = [self.word_to_ix.get(w, self.UNK_INDEX) for w in self.sentences[index]]
        tag = [self.tag_to_ix[t] for t in self.tags[index]]
        return torch.tensor(sent, dtype=torch.long), torch.tensor(tag, dtype=torch.long)

def create_collate_fn(pad_index):
    def collate_fn(batch):
        sentences = [item[0] for item in batch]
        tags = [item[1] for item in batch]
        # Đệm câu
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=pad_index)
        # Đệm nhãn
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_TAG_INDEX)
        return sentences_padded, tags_padded
    return collate_fn

# --- 2. Lớp Mô hình RNN (LSTM) ---

class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Dùng LSTM vì hiệu quả hơn RNN đơn giản
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True) 
        self.linear = nn.Linear(hid_dim, output_size)
        
    def forward(self, input_sentences):
        embedded = self.embedding(input_sentences)
        rnn_output, _ = self.rnn(embedded)
        tag_space = self.linear(rnn_output)
        return tag_space

# --- 3. Hàm Đánh giá ---

def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for sentences, tags in data_loader:
            sentences, tags = sentences.to(device), tags.to(device)
            tag_scores = model(sentences) 
            predictions = torch.argmax(tag_scores, dim=-1)
            
            non_padding_mask = (tags != PAD_TAG_INDEX)
            correct_predictions = (predictions == tags) & non_padding_mask
            
            total_correct += correct_predictions.sum().item()
            total_tokens += non_padding_mask.sum().item()

    return total_correct / total_tokens if total_tokens > 0 else 0

# --- CHƯƠNG TRÌNH CHÍNH ---

def run_ner_training():
    print("Bắt đầu quy trình huấn luyện NER...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tải và Tiền xử lý Dữ liệu
    dataset = load_dataset("conll2003")
    tag_names = dataset["train"].features["ner_tags"].feature.names

    def convert_tags_to_string(examples):
        examples["ner_tags_string"] = [tag_names[tag] for tag in examples["ner_tags"]]
        return examples
    dataset = dataset.map(convert_tags_to_string)
    
    # 2. Xây dựng Từ điển
    word_to_ix = defaultdict(lambda: len(word_to_ix))
    word_to_ix["<PAD>"] = 0
    word_to_ix["<UNK>"] = 1
    
    for sentence in dataset["train"]["tokens"]:
        for word in sentence:
            word_to_ix[word] 
            
    tag_to_ix = {tag: tag_names.index(tag) for tag in tag_names}

    # 3. Tạo DataLoader
    train_loader, valid_loader, test_loader = DataLoader(
        NERDataset(dataset["train"]["tokens"], dataset["train"]["ner_tags_string"], word_to_ix, tag_to_ix),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=create_collate_fn(word_to_ix["<PAD>"])
    ), DataLoader(
        NERDataset(dataset["validation"]["tokens"], dataset["validation"]["ner_tags_string"], word_to_ix, tag_to_ix),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=create_collate_fn(word_to_ix["<PAD>"])
    ), DataLoader(
        NERDataset(dataset["test"]["tokens"], dataset["test"]["ner_tags_string"], word_to_ix, tag_to_ix),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=create_collate_fn(word_to_ix["<PAD>"])
    )

    # 4. Khởi tạo Mô hình, Loss, Optimizer
    VOCAB_SIZE = len(word_to_ix)
    OUTPUT_SIZE = len(tag_to_ix)
    model = SimpleRNNForTokenClassification(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # ignore_index=PAD_TAG_INDEX (mặc định là -100) sẽ bỏ qua vị trí padding khi tính loss
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_TAG_INDEX) 

    # 5. Vòng lặp Huấn luyện
    print(f"\nBắt đầu huấn luyện {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for sentences, tags in train_loader:
            sentences, tags = sentences.to(device), tags.to(device)

            optimizer.zero_grad()
            tag_scores = model(sentences)
            
            # Cân chỉnh kích thước cho CrossEntropyLoss: (Batch*Seq_Len, Num_Tags) và (Batch*Seq_Len)
            loss = loss_function(tag_scores.view(-1, OUTPUT_SIZE), tags.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        
        # 6. Đánh giá
        val_accuracy = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Validation Acc: {val_accuracy:.4f}")
        print("\n--- ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST ---")
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Độ chính xác (Accuracy) trên Tập Test: {test_accuracy:.4f}")

if __name__ == '__main__':
    run_ner_training()