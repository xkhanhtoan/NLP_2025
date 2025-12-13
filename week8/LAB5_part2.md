
# Báo cáo Lab – Phân Loại Ý Định Người Dùng (Intent Classification)

## 1. Giới thiệu
Bài lab này thực hiện phân loại ý định người dùng trên bộ dữ liệu HWU. Bốn mô hình được xây dựng và so sánh:

1. TF-IDF + Logistic Regression
2. Word2Vec (mean embedding) + Dense
3. LSTM + Embedding từ đầu (scratch)
4. LSTM + Word2Vec pretrained

Mục tiêu:
- So sánh định lượng (accuracy, loss, F1)
- Phân tích định tính cho các câu khó
- Đánh giá khả năng xử lý chuỗi của LSTM
- Rút ra ưu và nhược điểm của từng mô hình

---

## 2. Phương pháp và mô hình

### 2.1 Tiền xử lý
- Chuyển chữ thường
- Xoá URL
- Xoá ký tự đặc biệt
- Tokenization, padding cho LSTM
- Huấn luyện hoặc load mô hình Word2Vec

### 2.2 Mô hình TF-IDF + Logistic Regression
Sử dụng tần suất từ (bag-of-words). Không dùng thứ tự từ. Mạnh với câu ngắn có keyword rõ ràng.

### 2.3 Word2Vec mean + Dense
Embedding của cả câu được tính bằng trung bình cộng embedding từng từ. Điều này làm mất hoàn toàn cấu trúc chuỗi.

### 2.4 LSTM + Embedding scratch
Embedding được khởi tạo ngẫu nhiên. LSTM học cả embedding lẫn cấu trúc câu.

### 2.5 LSTM + Word2Vec pretrained
Embedding khởi tạo từ mô hình Word2Vec đã huấn luyện. Kỳ vọng tốt hơn, nhưng phụ thuộc mức độ phù hợp vocab.

---

## 3. Kết quả định lượng

Dựa trên thực nghiệm:

| Mô hình | Accuracy | Nhận xét |
|--------|----------|----------|
| TF-IDF + Logistic Regression | 0.84 | Cao nhất, do dạng câu lệnh có nhiều keyword |
| LSTM scratch | 0.48 | Học được cấu trúc chuỗi |
| LSTM pretrained Word2Vec | 0.34 | Kém hơn scratch vì mismatch vocab |
| Word2Vec mean + Dense | 0.16 | Yếu nhất vì mất thứ tự từ |

Kết luận:
- TF-IDF rất mạnh trên dataset dạng câu lệnh.
- LSTM vượt trội Dense nhờ giữ thứ tự từ.
- Pretrained Word2Vec không phù hợp domain nên hoạt động kém.

---

## 4. Phân tích định tính

### Câu 1
"can you remind me to not call my mom"  
Nhãn: reminder_create

| Mô hình | Dự đoán | Nhận xét |
|--------|---------|----------|
| TF-IDF | Đúng | Bắt được keyword "remind" |
| Dense | Sai | Mất cấu trúc "remind me to ..." |
| LSTM scratch | Gần đúng | Học được mẫu câu mệnh lệnh |
| LSTM pretrained | Không ổn định | Phụ thuộc coverage |

LSTM hiểu được cấu trúc và phân biệt phủ định bên trong, Dense thì không.

---

### Câu 2
"is it going to be sunny or rainy tomorrow"  
Nhãn: weather_query

| Mô hình | Dự đoán | Nhận xét |
|--------|---------|----------|
| TF-IDF | Đúng | Từ khóa thời tiết rõ ràng |
| Dense | Sai | Mean embedding làm mờ chủ đề |
| LSTM scratch | Đúng | Hiểu chuỗi thời tiết |
| LSTM pretrained | Không ổn định | Phụ thuộc embedding có từ hay không |

---

### Câu 3
"find a flight from new york to london but not through paris"  
Nhãn: flight_search

| Mô hình | Dự đoán | Nhận xét |
|--------|---------|----------|
| TF-IDF | Có thể đúng | Từ "flight" mạnh |
| Dense | Sai | Không giữ được quan hệ "from - to - not via" |
| LSTM scratch | Đúng | Hiểu quan hệ xa giữa các từ |
| LSTM pretrained | Gần đúng | Nhưng phụ thuộc pretrained model |

Câu này thể hiện rõ nhất thế mạnh của LSTM vì chứa phụ thuộc xa và phủ định phức tạp.

---

## 5. Ưu và nhược điểm từng mô hình

### TF-IDF + LR
Ưu điểm:
- Hiệu quả cao trên dữ liệu nhiều keyword
- Dễ train, tránh overfitting

Nhược điểm:
- Không hiểu chuỗi và phủ định
- Không dùng được cho câu linh hoạt

### Word2Vec mean + Dense
Ưu điểm:
- Rất đơn giản, nhanh

Nhược điểm:
- Mất hoàn toàn thứ tự từ
- Kết quả rất thấp

### LSTM scratch
Ưu điểm:
- Hiểu chuỗi, phủ định, quan hệ xa
- Không phụ thuộc Word2Vec pretrained

Nhược điểm:
- Cần dữ liệu lớn hơn để tối ưu

### LSTM pretrained Word2Vec
Ưu điểm:
- Có thể tốt nếu pretrained khớp domain

Nhược điểm:
- Kém nếu vocab mismatch
- Phụ thuộc chất lượng Word2Vec

---

## 6. Kết luận
- LSTM vượt Dense nhờ giữ thứ tự từ và học được phụ thuộc xa.
- TF-IDF mạnh nhất trên HWU vì dataset nhiều keyword.
- Pretrained Word2Vec không phải lúc nào cũng tốt hơn scratch.
- Khi câu phức tạp, LSTM là mô hình thể hiện tốt nhất.

---

## 7. Nguồn tham khảo
1. Mikolov et al. “Word2Vec”, 2013.  
2. Hochreiter & Schmidhuber. “LSTM”, 1997.  
3. Jurafsky & Martin. Speech and Language Processing, 2023.  
4. HWU64 Intent Dataset.  
5. TensorFlow Documentation.  
6. Scikit-learn Documentation.  

