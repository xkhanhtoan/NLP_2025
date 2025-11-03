# Report: Thực hành PyTorch cơ bản (Lab 5)

## Mục tiêu bài tập
Bài tập này giúp sinh viên:
- Làm quen với **PyTorch**, thư viện học sâu phổ biến.  
- Hiểu rõ về **tensor** – cấu trúc dữ liệu cơ bản của PyTorch.  
- Biết cách tạo, thao tác và tính toán với tensor.  
- Làm quen với cơ chế **tự động tính gradient (autograd)** trong PyTorch.

---

## Nội dung chính của bài tập

### 1️.Khởi tạo Tensor

Bài tập bắt đầu bằng việc khởi tạo các tensor từ dữ liệu có sẵn hoặc sinh ngẫu nhiên:

```python
import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

Sau đó, tạo thêm các tensor khác:
- `torch.ones_like(x_data)` → tensor có cùng kích thước với `x_data` nhưng toàn giá trị 1.  
- `torch.rand_like(x_data)` → tensor ngẫu nhiên cùng kích thước.

Mục tiêu: giúp hiểu cách **tạo tensor từ dữ liệu**, **sao chép kích thước**, và **kiểm tra kiểu dữ liệu (`dtype`)**.

---

### 2.Kiểm tra thuộc tính của Tensor

Ví dụ lệnh:
```python
print(x_rand.shape)
print(x_rand.dtype)
print(x_rand.device)
```

Giúp người học nắm:
- **Shape** → kích thước tensor (hàng, cột)
- **Datatype** → kiểu dữ liệu (float, int,…)
- **Device** → nơi lưu trữ tensor (CPU hoặc GPU)

---

### 3.Các phép toán cơ bản trên Tensor

```python
add_result = x_data + x_data
multiply_result = x_data * 5
transpose_result = x_data @ x_data.T
```

- **Cộng, nhân vô hướng, nhân ma trận** được thực hiện tương tự NumPy.  
- Giúp người học hiểu cách thực hiện **phép tính vector/matrix** trên GPU dễ dàng.

---

### 4.Tính toán với Gradient

Phần tiếp theo giới thiệu **Autograd – cơ chế tự động đạo hàm của PyTorch**.

```python
x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y * 3
z.backward()
```

- Khi `requires_grad=True`, PyTorch sẽ **theo dõi tất cả phép toán** để có thể tính đạo hàm ngược.  
- Lệnh `z.backward()` tự động **tính gradient của z theo x**.  
- Gradient được lưu tại `x.grad`.

### 5.Mô hình (Model)

Phần này trình bày mô hình **`MyFirstModel`**, một mạng nơ-ron đơn giản được xây dựng bằng **PyTorch** nhằm minh họa quy trình truyền xuôi (forward pass) trong học sâu.

Mô hình bao gồm ba thành phần chính:
- **Embedding layer:** chuyển các chỉ số từ (token index) thành vector thực cố định, giúp biểu diễn ý nghĩa của từ trong không gian liên tục.  
- **Hidden layer (Linear + ReLU):** biến đổi các vector embedding thành đặc trưng ẩn, đồng thời bổ sung phi tuyến tính để mô hình có thể học được các quan hệ phức tạp hơn.  
- **Output layer:** ánh xạ đặc trưng ẩn sang đầu ra có kích thước tương ứng với nhiệm vụ (ví dụ: số lớp phân loại).

Dữ liệu đầu vào là các chỉ số từ, đi qua các tầng theo thứ tự embedding → linear → kích hoạt → output để tạo ra đầu ra cuối cùng cho mỗi token.  
Trong phạm vi bài lab, mô hình được dùng để kiểm tra **quá trình truyền dữ liệu (forward)** và **kích thước tensor** qua từng lớp, chưa thực hiện huấn luyện với hàm mất mát.  

Mô hình này là bước khởi đầu giúp người học nắm được cách tổ chức, định nghĩa và vận hành một mạng nơ-ron trong PyTorch, tạo nền tảng cho việc xây dựng các mô hình ngôn ngữ phức tạp hơn như LSTM hoặc Transformer.
