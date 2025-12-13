# Lab 5 (Final): Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER)

## 1. Mục tiêu
Trong bài thực hành này, chúng tôi xây dựng một mô hình dựa trên Mạng Nơ-ron Hồi quy (RNN/LSTM) cho bài toán Nhận dạng Thực thể Tên (Named Entity Recognition – NER).
Sau khi hoàn thành bài lab, mô hình có khả năng:
- Tải và tiền xử lý dữ liệu NER từ thư viện Hugging Face.
- Xây dựng vocabulary cho từ và nhãn NER.
- Triển khai Dataset và DataLoader tùy chỉnh trong PyTorch.
- Xây dựng và huấn luyện mô hình LSTM cho bài toán token classification.
- Đánh giá hiệu năng mô hình trên bộ dữ liệu CoNLL-2003.

---

## 2. Bộ dữ liệu: CoNLL 2003
Chúng tôi sử dụng bộ dữ liệu **CoNLL-2003**, một benchmark tiêu chuẩn cho bài toán NER.
Dữ liệu được tải thông qua thư viện `datasets`:

```python
datasets.load_dataset("conll2003", trust_remote_code=True)
```

Bộ dữ liệu bao gồm ba tập:
- Train
- Validation
- Test

Nhãn được gán theo chuẩn **IOB** (B-XXX, I-XXX, O).

---

## 3. Các bước thực hiện

### Task 1: Tải và Tiền xử lý Dữ liệu
- Trích xuất danh sách token và nhãn NER từ tập train.
- Chuyển đổi nhãn từ dạng số sang dạng chuỗi (B-PER, I-LOC, O, …).
- Xây dựng hai từ điển:
  - `word_to_ix`: ánh xạ từ → chỉ số, bổ sung `<PAD>` và `<UNK>`.
  - `tag_to_ix`: ánh xạ nhãn NER → chỉ số.

### Task 2: Tạo Dataset và DataLoader
- Xây dựng lớp `NERDataset` kế thừa `torch.utils.data.Dataset`.
- Mỗi mẫu dữ liệu trả về tensor từ và tensor nhãn tương ứng.
- Sử dụng hàm `collate_fn` để padding:
  - Token padding: `<PAD>`
  - Nhãn padding: `-100` (được ignore trong loss).

### Task 3: Xây dựng Mô hình RNN
Mô hình bao gồm:
1. `nn.Embedding` để ánh xạ từ sang vector.
2. `nn.LSTM` để học ngữ cảnh chuỗi.
3. `nn.Linear` để dự đoán nhãn cho từng token.

### Task 4: Huấn luyện Mô hình
- Optimizer: Adam.
- Loss function: `CrossEntropyLoss(ignore_index=-100)`.
- Huấn luyện trong 5 epochs.
- Theo dõi loss và accuracy trên tập validation.

### Task 5: Đánh giá Mô hình
- Đánh giá mô hình trên tập validation và test.
- Accuracy chỉ được tính trên các token không phải padding.

---

## 4. Kết quả Thực nghiệm

### Kết quả huấn luyện
| Epoch | Validation Accuracy | Test Accuracy |
|------|---------------------|---------------|
| 1 | 0.8645 | 0.8537 |
| 2 | 0.9052 | 0.8850 |
| 3 | 0.9196 | 0.8964 |
| 4 | 0.9258 | 0.9033 |
| 5 | 0.9249 | 0.8970 |

Độ chính xác cao nhất trên tập test đạt **khoảng 90%**.

---

## 5. Ví dụ dự đoán câu mới
- **Câu**: “VNU University is located in Hanoi”
- **Dự đoán**:
  - VNU → B-ORG
  - University → I-ORG
  - is → O
  - located → O
  - in → O
  - Hanoi → B-LOC

---

## 6. Nhận xét và Phân tích
- Mô hình LSTM học được ngữ cảnh chuỗi, giúp cải thiện kết quả so với các mô hình không xét ngữ cảnh.
- Accuracy ~90% là kết quả tốt đối với mô hình LSTM cơ bản, chưa sử dụng BiLSTM hay CRF.
- Một số lỗi vẫn xảy ra với các thực thể hiếm hoặc thực thể dài.

---

## 7. Khó khăn và Giải pháp
- **Padding trong token classification**: được xử lý bằng `ignore_index` trong hàm loss.
- **Dataset có custom code**: sử dụng `trust_remote_code=True` khi load dữ liệu.
- **Thời gian huấn luyện**: giới hạn số epoch để phù hợp với tài nguyên máy cá nhân.

---

## 8. Kết luận
Bài lab đã xây dựng thành công mô hình RNN/LSTM cho bài toán NER trên CoNLL-2003.
Kết quả cho thấy mô hình có khả năng nhận dạng thực thể tên hiệu quả và minh họa rõ sức mạnh của mô hình chuỗi trong NLP.

---

## 9. Tài liệu tham khảo
- CoNLL-2003 Dataset
- Hugging Face Datasets Documentation
- PyTorch Documentation
