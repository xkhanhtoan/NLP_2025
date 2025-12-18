# Lab X – Tổng quan bài toán Text To Speech (TTS)

## 1. Bối cảnh
Trong bối cảnh công nghệ phát triển nhanh chóng, khả năng tự học và tự nghiên cứu trở thành kỹ năng then chốt đối với sinh viên ngành Công nghệ Thông tin và Trí tuệ Nhân tạo. Với sự hỗ trợ của Internet, các hệ thống tìm kiếm và AI/Agent, việc tiếp cận tri thức và nghiên cứu chuyên sâu trở nên dễ dàng hơn bao giờ hết.

Trong nội dung bổ sung của tuần 12, sinh viên đóng vai trò như một nhà nghiên cứu, tìm hiểu tổng quan về bài toán Text To Speech (TTS) – chuyển đổi văn bản thành giọng nói. Yêu cầu chính của bài tập là khảo sát tình hình nghiên cứu, các hướng tiếp cận hiện tại, cũng như phân tích ưu và nhược điểm của từng hướng.

## 2. Tổng quan bài toán Text To Speech (TTS)
Text To Speech (TTS) là bài toán chuyển đổi văn bản thành tín hiệu âm thanh sao cho giọng nói tạo ra dễ nghe, tự nhiên và phù hợp với ngữ cảnh. TTS là thành phần quan trọng trong trợ lý ảo, hệ thống đọc văn bản, tổng đài tự động và sản xuất nội dung số.

## 3. Các hướng tiếp cận chính

### Level 1 – Rule-based TTS
Dựa trên luật ngữ âm và từ điển phát âm.
- Ưu điểm: nhanh, ít tài nguyên
- Nhược điểm: giọng nói kém tự nhiên
- Phù hợp: thiết bị nhúng, ngôn ngữ ít dữ liệu

### Level 2 – Deep Learning TTS
Sử dụng mô hình học sâu (Tacotron, FastSpeech, WaveNet).
- Ưu điểm: giọng nói tự nhiên
- Nhược điểm: cần dữ liệu và tài nguyên
- Phù hợp: trợ lý ảo, sản phẩm thương mại

### Level 3 – Few-shot / Zero-shot TTS
Chỉ cần vài giây audio mẫu để clone giọng nói.
- Ưu điểm: linh hoạt, cá nhân hóa cao
- Nhược điểm: model lớn, rủi ro deepfake
- Phù hợp: nghiên cứu, sáng tạo nội dung

## 4. So sánh
| Level | Tự nhiên | Tài nguyên | Cá nhân hóa |
|------|----------|------------|-------------|
| 1 | Thấp | Rất thấp | Không |
| 2 | Cao | Trung bình | Có |
| 3 | Rất cao | Cao | Rất tốt |

## 5. Thách thức
- Tốc độ sinh âm thanh
- Chi phí tính toán
- Đa ngôn ngữ
- Biểu đạt cảm xúc
- Đạo đức và deepfake

## 6. Pipeline nghiên cứu hiện đại
Các nghiên cứu hiện nay thường tách pipeline Text → Acoustic → Vocoder, kết hợp pre-trained model và fine-tuning, dùng speaker embedding và distillation để tối ưu hiệu suất.

## 7. Đạo đức nghiên cứu
Cần nhúng watermark, xác thực nguồn audio và hạn chế lạm dụng công nghệ TTS.

## 8. Kết luận
Không có phương pháp TTS tối ưu cho mọi bài toán. Việc lựa chọn giải pháp phụ thuộc vào nhu cầu, dữ liệu và tài nguyên.

## 9. Tài liệu tham khảo
- Tacotron 2
- WaveNet
- FastSpeech
- VALL-E
- Hugging Face TTS
