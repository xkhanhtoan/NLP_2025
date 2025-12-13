# README.md – Lab 5: Phân loại văn bản

## 1. Giới thiệu
Lab 5 tập trung vào việc xây dựng và đánh giá một hệ thống **phân loại văn bản (Text Classification)** bằng **Scikit-learn** và **Apache Spark**.
Bài lab bao gồm các task: cài đặt mô hình, kiểm thử, chạy ví dụ Spark và thử nghiệm cải tiến mô hình.

---

## 2. Cấu trúc thư mục
src/
 └─ models/
     ├─ my_vectorizer.py
     ├─ my_vectorizer_v2.py
     ├─ text_classification.py
     ├─ text_classification_nb.py
     └─ text_classification_nn.py

test/
 ├─ lab5_test.py
 ├─ lab5_improvement_test.py
 └─ lab5_spark_sentiment_analysis.py

---

## 3. Task 1 – Cài đặt TextClassifier
Một hệ thống phân loại văn bản cơ bản được cài đặt bằng Scikit-learn:
- Vectorizer: TF-IDF tự cài đặt
- Mô hình: Logistic Regression
- Đánh giá: Accuracy, Precision, Recall, F1-score

Lớp `TextClassification` cung cấp các hàm `fit`, `predict` và `evaluate` để huấn luyện và đánh giá mô hình.

---

## 4. Task 2 – Kiểm thử cơ bản
File `test/lab5_test.py` thực hiện:
- Khai báo tập dữ liệu văn bản và nhãn
- Chia dữ liệu thành tập train/test theo tỉ lệ 80/20
- Huấn luyện mô hình Logistic Regression
- Dự đoán và in ra các chỉ số đánh giá

Task này đóng vai trò là baseline model để so sánh với các phương pháp cải tiến.

---

## 5. Task 3 – Phân tích cảm xúc với Spark
Một pipeline phân loại văn bản bằng Spark ML được xây dựng trong file `test/lab5_spark_sentiment_analysis.py`.

Pipeline bao gồm:
- Tokenizer
- StopWordsRemover
- HashingTF
- IDF
- Logistic Regression

Dataset được lấy từ HuggingFace: `zeroshot/twitter-financial-news-sentiment`.

---

## 6. Task 4 – Thử nghiệm cải tiến mô hình
Các phương pháp cải tiến:
1. Cải thiện preprocessing và chọn đặc trưng
2. Naive Bayes
3. Neural Network (MLP)

File `lab5_improvement_test.py` được dùng để so sánh với mô hình baseline.

---

## 7. Kết quả và phân tích
- Logistic Regression baseline dùng làm mốc so sánh
- Preprocessing giúp mô hình ổn định hơn
- Naive Bayes phù hợp với dữ liệu nhỏ
- Neural Network không vượt trội do dữ liệu hạn chế

---

## 8. Khó khăn và cách giải quyết
Khi chạy Spark trên Windows với Python 3.12 xuất hiện lỗi Python worker failed to connect back.
Đây là lỗi môi trường, không phải lỗi logic code.
Pipeline Spark được cài đặt đúng và logic xử lý được hiểu rõ.

---

## 9. Hướng dẫn chạy code
python test/lab5_test.py
python test/lab5_improvement_test.py
python test/lab5_spark_sentiment_analysis.py

---

## 10. Tài liệu tham khảo
- https://scikit-learn.org
- https://spark.apache.org/docs/latest/ml-guide.html
- https://huggingface.co/datasets
