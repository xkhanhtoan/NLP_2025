# Báo cáo Lab 17 – Xây dựng Spark NLP Pipeline

## 1. Tổng quan và Mục tiêu

Bài Lab này nhằm giúp sinh viên làm quen với việc xây dựng **Spark ML Pipeline** để xử lý ngôn ngữ tự nhiên (NLP).  
Mục tiêu là thực hiện toàn bộ quy trình từ đọc dữ liệu văn bản, tiền xử lý (tokenization, stopword removal), vector hóa (TF-IDF hoặc Word2Vec), huấn luyện Logistic Regression, chuẩn hóa vector, đến việc tìm các văn bản tương tự nhất.

---

## 2. Các bước triển khai

### Bước 1 – Đọc dữ liệu và khởi tạo DataFrame
- Đọc dữ liệu **C4 dataset** vào Spark DataFrame.  
- Áp dụng biến `limitDocuments` để giới hạn số lượng tài liệu khi cần thử nghiệm.

```scala
val limitDocuments = 1000
val df = spark.read.json("path/to/c4_sample.json").limit(limitDocuments)
```

---

### Bước 2 – Tokenization (Tách từ)
- Thử nghiệm cả hai lựa chọn:
  - `Tokenizer`: chia tách đơn giản theo khoảng trắng.
  - `RegexTokenizer`: loại bỏ được ký tự đặc biệt và dấu câu.  
- Trong pipeline cuối, chọn `RegexTokenizer` vì kết quả chính xác hơn.

---

### Bước 3 – Xử lý Stop Words
- Dùng `StopWordsRemover` để loại bỏ các từ dừng phổ biến, giúp giảm nhiễu trong vector TF-IDF và Word2Vec.

---

### Bước 4 – Vector hóa văn bản
#### (a) TF-IDF
- Áp dụng `HashingTF` và `IDF` để chuyển token thành vector đặc trưng.  
- Thử 2 cấu hình: `numFeatures = 20000` và `numFeatures = 1000`.
- Khi giảm còn 1000, xuất hiện hash collision nhưng xử lý nhanh hơn.

#### (b) Word2Vec
- Dùng `Word2Vec` (vector size 100, minCount = 2) để sinh embedding dense.  
- Kết hợp Logistic Regression để thử nghiệm phân loại.

---

### Bước 5 – Chuẩn hóa vector
- Thêm bước **Normalizer** (chuẩn L2) để cân bằng các vector trước khi đo độ tương đồng.

```scala
val normalizer = new Normalizer()
  .setInputCol("features")
  .setOutputCol("normFeatures")
  .setP(2.0)
```

---

### Bước 6 – Huấn luyện Logistic Regression
- Dataset không có nhãn → tạo cột `label` giả để kiểm thử.
- Dùng `LogisticRegression(maxIter=20, regParam=0.01)` với tham số điều chỉnh hợp lý.

---

### Bước 7 – Tìm tương đồng giữa các văn bản
- Tính cosine similarity giữa văn bản đầu tiên và toàn bộ tập.
- Trả về **Top K văn bản tương đồng nhất** (K = 5).

---

### Bước 8 – Lưu và log kết quả
- Ghi log chi tiết thời gian và thông tin pipeline.

| File | Mô tả |
|------|-------------|
| `lab17_metrics.log` | TF-IDF pipeline |
| `lab17_metrics_word2vec_lr.log` | Word2Vec + Logistic Regression |

---

## 3. Cách chạy và log kết quả

Chạy trực tiếp với **sbt**:
```bash
sbt "runMain com.quangviet.spark.Lab17_NLPPipeline"
sbt "runMain com.quangviet.spark.Lab17_NLPPipeline_Word2Vec_lr"
```

Kết quả được sinh ra trong thư mục:
- `./log/` : file log hiệu năng.
- `./result/` : kết quả pipeline.

---

## 4. Kết quả và Phân tích

### TF-IDF Pipeline
- Thời gian fitting: **3.5–4.5s**  
- Transformation: **~2s**  
- Vocabulary: **31k–46k**  
- Giảm `numFeatures` giúp chạy nhanh nhưng giảm chất lượng.

### Word2Vec + Logistic Regression
Trích xuất từ log `lab17_metrics_word2vec_lr.log`:

| Thông số | Giá trị |
|-----------|----------|
| Vector size | 100 |
| Training Accuracy | 82.10% |
| Test Accuracy | **85.80%** |
| Thời gian huấn luyện | 30.88s |

**Word2Vec cho kết quả tốt hơn TF-IDF**, nhưng tốn nhiều tài nguyên hơn.

---

## 5. Khó khăn và Giải pháp

| Vấn đề | Giải pháp |
|---------|------------|
| Dataset không có nhãn | Tạo label giả để demo LR |
| Hash collision khi giảm numFeatures | Giữ mức 20000 cho pipeline chính |
| Thời gian huấn luyện Word2Vec dài | Giảm iter và vectorSize |

---

## 6. Tài liệu tham khảo
- Apache Spark MLlib Documentation: https://spark.apache.org/docs/latest/ml-guide.html  
- Word2Vec API (Spark ML): https://spark.apache.org/docs/latest/ml-features#word2vec  
- Dataset: C4 (Colossal Clean Crawled Corpus)
