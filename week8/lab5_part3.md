# Part 3: Part-of-Speech Tagging với RNN

## 1. Giới thiệu
Trong phần này, chúng tôi xây dựng một mô hình **RNN** để giải quyết bài toán **Part-of-Speech (POS) Tagging**, trong đó mỗi token trong câu được gán một nhãn ngữ pháp (noun, verb, adjective, …). Mục tiêu là kiểm nghiệm khả năng của RNN trong việc mô hình hóa ngữ cảnh chuỗi cho bài toán phân loại token.

---

## 2. Chuẩn bị dữ liệu (Task 1)

### 2.1 Bộ dữ liệu
Sử dụng bộ dữ liệu **UD-English (Universal Dependencies – English EWT)** ở định dạng **CoNLL-U**.  
Các cột sử dụng:
- FORM (cột 2): từ
- UPOS (cột 4): nhãn POS

### 2.2 Tiền xử lý
- Bỏ qua dòng comment và dòng trống  
- Chỉ giữ các dòng hợp lệ  
- Gom token thành câu

### 2.3 Xây dựng Vocabulary
- `word_to_ix`: <PAD>=0, <UNK>=1  
- `tag_to_ix`: <PAD>=0, các nhãn POS bắt đầu từ 1  

**Kết quả:**
- Vocabulary size: **15630**
- Số nhãn POS: **18**

---

## 3. Xây dựng mô hình RNN (Task 2)

Pipeline mô hình:
```
Embedding → RNN → Linear
```

- Embedding: biểu diễn từ dạng vector
- RNN: học ngữ cảnh chuỗi
- Linear: dự đoán nhãn POS cho từng token

Loss function: `CrossEntropyLoss(ignore_index=0)`

---

## 4. Huấn luyện và Đánh giá (Task 3)

### 4.1 Thiết lập
- Optimizer: Adam  
- Learning rate: 0.001  
- Batch size: 32  
- Epochs: 5  

### 4.2 Kết quả

| Epoch | Loss | Dev Accuracy |
|------|------|--------------|
| 1 | 405.2724 | 0.6943 |
| 2 | 219.9034 | 0.7463 |
| 3 | 166.6726 | 0.7645 |
| 4 | 134.1091 | 0.7853 |
| 5 | 111.1755 | **0.8000** |

---

## 5. Phân tích kết quả
- Accuracy tăng đều qua các epoch
- RNN học được ngữ cảnh cơ bản
- Hạn chế với chuỗi dài do không dùng LSTM/GRU

---

## 6. Khó khăn và Giải pháp
**Khó khăn**
- File CoNLL-U chứa dòng đặc biệt
- Lỗi mismatch nhãn và output
- Dev set không chuẩn

**Giải pháp**
- Lọc dòng khi đọc dữ liệu
- Chuẩn hóa padding và nhãn
- Dùng tập train làm dev để kiểm tra pipeline

---

## 7. Kết luận
Mô hình RNN cho POS Tagging được triển khai thành công, đạt **80% accuracy** trên tập dev, là nền tảng cho các mô hình chuỗi nâng cao hơn như LSTM hoặc BiLSTM.
