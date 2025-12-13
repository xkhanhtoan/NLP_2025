# BÁO CÁO LAB 5: PHÂN LOẠI VĂN BẢN

## 1. Giới thiệu
Lab 5 tập trung vào việc xây dựng, đánh giá và cải tiến hệ thống phân loại văn bản (Text Classification)
sử dụng Scikit-learn và Apache Spark. Mục tiêu của bài lab là hiểu quy trình xử lý văn bản từ tiền xử lý,
trích xuất đặc trưng, huấn luyện mô hình cho đến đánh giá hiệu quả.

---

## 2. Các bước cài đặt và triển khai

Quy trình triển khai hệ thống gồm các bước sau:

1. Tiền xử lý dữ liệu  
   - Sử dụng RegexTokenizer để chuẩn hóa văn bản (chuyển chữ thường, loại bỏ ký tự đặc biệt).  
   - Tách văn bản thành các token.

2. Biểu diễn đặc trưng  
   - Áp dụng TF-IDF Vectorizer để chuyển văn bản sang vector số.  
   - Giảm ảnh hưởng của các từ xuất hiện quá thường xuyên.

3. Huấn luyện mô hình  
   - Baseline: Logistic Regression.  
   - Mô hình cải tiến: Logistic Regression (tuned), Naive Bayes, Neural Network.

4. Đánh giá mô hình  
   - Sử dụng các chỉ số: Accuracy, Precision, Recall và F1-score.

---

## 3. Hướng dẫn chạy code

Baseline model:
python -m test.lab5_test

Mô hình cải tiến:
python -m test.lab5_test_ver2

Ví dụ Spark:
python -m test.lab5_spark_sentiment_analysis

---

## 4. Kết quả và phân tích

### 4.1 Baseline: Logistic Regression

Tập test gồm 2 câu văn bản:
y_test = [1, 0]  
y_pred = [0, 1]

Confusion Matrix:
[[0 1]
 [1 0]]

Kết quả:
- Accuracy: 0.00  
- Precision: 0.00  
- Recall: 0.00  
- F1-score: 0.00  

Nhận xét:
Mô hình dự đoán sai toàn bộ tập test do tập dữ liệu quá nhỏ và sự không ổn định của đặc trưng TF-IDF.

---

### 4.2 Mô hình cải tiến

Kết quả từ lab5_test_ver2:
- Improved Logistic Regression: Accuracy = 0.5  
- Naive Bayes: Accuracy = 0.5  
- Neural Network: Accuracy = 0.0  

So sánh:
Các mô hình Logistic Regression cải tiến và Naive Bayes cho kết quả tốt hơn baseline.
Neural Network không hiệu quả do dữ liệu huấn luyện quá ít.

---

## 5. Khó khăn và hướng giải quyết

Trong quá trình thực hiện, việc chạy Apache Spark trên Windows với Python 3.12 gặp lỗi:
Python worker failed to connect back.

Đây là lỗi môi trường (Java, Spark, Python), không phải lỗi logic chương trình.
Giải pháp là mô tả chi tiết Spark ML Pipeline thay vì yêu cầu chạy Spark thành công.

---

## 6. Tài liệu tham khảo

- Scikit-learn Documentation: https://scikit-learn.org/
- Apache Spark MLlib Documentation: https://spark.apache.org/
- HuggingFace Datasets: https://huggingface.co/datasets
