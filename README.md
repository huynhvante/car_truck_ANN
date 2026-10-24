# Car vs Truck Classification using CNN

Dự án xây dựng mô hình học sâu để **phân loại ảnh phương tiện thành hai lớp: `car` (ô tô) và `truck` (xe tải)**. Mô hình được xây dựng bằng TensorFlow/Keras, xử lý ảnh bằng OpenCV và có thể đánh giá ảnh đầu vào bằng chương trình Python.
Datasets: https://www.kaggle.com/datasets/ryanholbrook/car-or-truck

## ✨ Tính năng

- Huấn luyện mô hình CNN để phân loại ảnh ô tô và xe tải.
- Đọc và tiền xử lý ảnh bằng OpenCV.
- Thay đổi kích thước ảnh đầu vào về `64 × 64` pixel.
- Chia dữ liệu thành tập huấn luyện, validation và kiểm thử.
- Lưu mô hình đã huấn luyện để sử dụng cho dự đoán.
- Chạy dự đoán trên các ảnh trong thư mục kiểm tra và hiển thị nhãn cùng độ tin cậy.

## 🧰 Công nghệ sử dụng

- **Python**
- **TensorFlow / Keras** — xây dựng, huấn luyện và tải mô hình.
- **OpenCV** — đọc, thay đổi kích thước và hiển thị ảnh.
- **NumPy** — xử lý mảng ảnh.
- **scikit-learn** — các tiện ích chia tập dữ liệu (được import trong mã nguồn).
- **Matplotlib**, **tqdm**, **pandas** — các tiện ích hỗ trợ được import trong mã huấn luyện.

## 📁 Cấu trúc dự án

```text
car_truck_ANN/
├── Model/             # Thư mục mô hình đã lưu (nếu có)
├── Model1/            # Thư mục mô hình/dữ liệu bổ sung (cần kiểm tra nội dung thực tế)
├── gui3.py            # Chạy dự đoán trên ảnh trong thư mục kiểm tra
├── train_model.py     # Chuẩn bị dữ liệu, xây dựng và huấn luyện CNN
└── README.md
```

> Nội dung thực tế của `Model/` và `Model1/` phụ thuộc vào các tệp được lưu trong repository. Hãy kiểm tra để xác định thư mục nào là mô hình dùng khi dự đoán.

## 🖼️ Chuẩn bị dữ liệu

Trong `train_model.py`, đường dẫn dữ liệu được cấu hình qua biến `DATADIR`, còn hai lớp được khai báo là `car` và `truck`.

Sắp xếp tập dữ liệu theo cấu trúc:

```text
Datasets/
├── car/
│   ├── car_001.jpg
│   ├── car_002.jpg
│   └── ...
└── truck/
    ├── truck_001.jpg
    ├── truck_002.jpg
    └── ...
```

Ảnh có thể ở định dạng được OpenCV đọc hỗ trợ. Nên loại bỏ ảnh hỏng, ảnh không liên quan và dữ liệu trùng lặp trước khi huấn luyện.

## ⚙️ Cài đặt

Khuyến nghị sử dụng môi trường ảo Python.

### Windows (PowerShell)

```powershell
git clone https://github.com/huynhvante/car_truck_ANN.git
cd car_truck_ANN

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install tensorflow opencv-python numpy scikit-learn matplotlib pandas tqdm
```

Nếu môi trường của bạn không cài được một phiên bản TensorFlow cụ thể, hãy chọn phiên bản TensorFlow tương thích với phiên bản Python và hệ điều hành đang dùng.

## 🏋️ Huấn luyện mô hình

1. Chuẩn bị dữ liệu theo cấu trúc ở trên.
2. Mở `train_model.py`.
3. Sửa `DATADIR` thành đường dẫn đến thư mục `Datasets` trên máy của bạn.
4. Kiểm tra đường dẫn lưu mô hình ở cuối tệp và sửa thành vị trí mong muốn.
5. Chạy:

```powershell
python train_model.py
```

Trong mã nguồn hiện tại, ảnh được đưa về kích thước `64 × 64 × 3`, mô hình CNN sử dụng các lớp tích chập, Batch Normalization, Max Pooling, Dropout và các lớp Dense. Mã nguồn đặt `epochs = 10` và `batch_size = 32`.

Sau huấn luyện, chương trình đánh giá mô hình trên tập test. Kết quả phụ thuộc vào chất lượng, số lượng và độ đa dạng của dữ liệu.

## 🔍 Dự đoán ảnh

`gui3.py` tải mô hình đã lưu, đọc các ảnh trong thư mục kiểm tra, thay đổi kích thước ảnh về `64 × 64`, sau đó dự đoán một trong hai nhãn `car` hoặc `truck`.

Trước khi chạy:

1. Sửa đường dẫn tải mô hình trong `gui3.py` để trỏ đến thư mục mô hình đã được huấn luyện.
2. Sửa biến `path` để trỏ đến thư mục chứa ảnh cần kiểm tra.
3. Đảm bảo mô hình được tải có cùng định dạng đầu vào và thứ tự nhãn với mã dự đoán.

Chạy:

```powershell
python gui3.py
```

Chương trình in kết quả dự đoán ra terminal và hiển thị ảnh cùng nhãn dự đoán khi mức xác suất lớn nhất vượt ngưỡng `0.7`.

## 🧠 Tổng quan mô hình

- **Bài toán:** phân loại ảnh nhị phân.
- **Các lớp đầu ra:** `car`, `truck`.
- **Kích thước đầu vào:** `64 × 64 × 3`.
- **Kiến trúc:** CNN với các lớp Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten và Dense.
- **Optimizer:** Adam.
- **Số epoch cấu hình trong mã:** 10.
- **Batch size cấu hình trong mã:** 32.

## ⚠️ Lưu ý

- Mã nguồn hiện có các đường dẫn tuyệt đối phụ thuộc vào máy phát triển ban đầu. Cần thay chúng bằng đường dẫn hợp lệ trên máy của bạn trước khi chạy.
- Thứ tự nhãn trong mã dự đoán phải khớp với cách mã hóa nhãn khi huấn luyện.
- Ngưỡng `0.7` chỉ là ngưỡng quyết định được đặt trong chương trình, không phải cam kết rằng dự đoán đạt độ chính xác 70%.
- Để đánh giá khách quan, nên kiểm tra precision, recall, F1-score, confusion matrix và hiệu năng trên ảnh chưa từng xuất hiện trong tập huấn luyện.
- Các thư mục `Model/` và `Model1/` cần được kiểm tra để xác nhận mô hình nào được dùng trong quá trình dự đoán.

## 🚀 Hướng phát triển

- Chuẩn hóa đường dẫn bằng `pathlib` hoặc `os.path`.
- Tách tập train/validation/test rõ ràng và cố định random seed.
- Bổ sung biểu đồ accuracy/loss và báo cáo các chỉ số phân loại.
- Xây dựng giao diện để chọn ảnh và hiển thị kết quả dự đoán.
- Bổ sung kiểm tra ảnh đầu vào không hợp lệ và thông báo lỗi rõ ràng.

## 📄 License

Chưa xác định giấy phép từ các tệp hiện có trong repository. Nếu muốn cho phép người khác sử dụng, chỉnh sửa hoặc phân phối dự án, hãy bổ sung tệp `LICENSE` phù hợp.
