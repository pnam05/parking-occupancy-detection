# Smart Parking Management System

Hệ thống quản lý bãi đỗ xe thông minh sử dụng thị giác máy tính (**OpenCV**) và học sâu (**Deep Learning - MobileNetV3**). Hệ thống có khả năng tự động nhận diện trạng thái còn trống hoặc đã có xe của từng ô đỗ dựa trên vùng nhận diện (ROI) được thiết lập trước.

## Demo Hệ Thống

---

## Các tính năng nổi bật

- **ROI Setup Tool**: Công cụ thiết lập vùng đỗ xe linh hoạt hỗ trợ phóng to/thu nhỏ (Zoom) và di chuyển góc nhìn (Pan) để xử lý camera ở vị trí xa hoặc có độ phân giải cao.
- **Data Annotation**: Quy trình thu thập và gán nhãn dữ liệu trực tiếp từ nguồn video, cho phép tạo bộ dữ liệu huấn luyện nhanh chóng.
- **Image Registration**: Sử dụng thuật toán **ORB** và ma trận **Homography** để ổn định khung hình, đảm bảo hệ thống hoạt động chính xác ngay cả khi camera bị rung lắc.
- **Lightweight Model**: Sử dụng kiến trúc **MobileNetV3-Small** giúp tối ưu tốc độ xử lý (inference) và tiết kiệm tài nguyên phần cứng.

---

## Cấu trúc thư mục

- **draw_roi.py**: Công cụ xác định vị trí các ô đỗ xe trên khung hình video.
- **collect_data.py**: Script cắt ảnh từ video và gán nhãn dữ liệu để chuẩn bị cho việc huấn luyện.
- **train_classification.py**: Thực hiện huấn luyện mô hình phân loại trạng thái ô đỗ xe.
- **main.py**: Chương trình chính thực hiện nhận diện và hiển thị kết quả thời gian thực.
- **util.py**: Chứa các hàm hỗ trợ về xử lý hình học và cắt ảnh theo đa giác.

---

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

Sử dụng pip để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 2. Thiết lập vùng đỗ xe (ROI)

Chạy công cụ vẽ để xác định các vị trí ô đỗ xe:

```bash
python draw_roi.py
```

- **Chuột trái**: Click chọn 4 góc của một ô đỗ xe để tạo 1 vùng ROI.
- **Chuột phải và kéo**: Di chuyển góc nhìn khi đang phóng to.
- **Lăn chuột hoặc phím +/-**: Phóng to hoặc thu nhỏ khung hình.
- **Phím S**: Lưu danh sách tọa độ vào tệp rois.json.

### 3. Thu thập dữ liệu gán nhãn

Tiến hành thu thập ảnh mẫu cho từng trạng thái:

```bash
python collect_data.py
```

- **Click chuột vào ô đỗ**: Chọn ô cần gán nhãn.
- **Phím E**: Gán nhãn là Trống (Empty).
- **Phím O**: Gán nhãn là Có xe (Occupied).
- **Phím S**: Lưu các ảnh đã gán nhãn và tự động nhảy qua 30 khung hình tiếp theo.
- **Phím A**: Tự động lưu tất cả các ô trong khung hình hiện tại là Trống.

### 4. Huấn luyện mô hình

Chạy script huấn luyện sau khi đã chuẩn bị đủ dữ liệu trong thư mục **dataset/**:

```bash
python train_classification.py
```

Mô hình tốt nhất sẽ được lưu tại đường dẫn **weights/best.pth**.

### 5. Triển khai hệ thống

Khởi chạy hệ thống nhận diện thực tế trên luồng video:

```bash
python main.py
```

## Cơ chế hoạt động của hệ thống chính

- **Tính năng Alignment (Chống rung)**: Cứ sau mỗi 90 khung hình, hệ thống sẽ thực hiện so sánh các điểm đặc trưng giữa khung hình hiện tại và ảnh gốc (reference_frame.jpg) bằng thuật toán ORB để tính toán ma trận dịch chuyển.

- **Xử lý đa luồng (Threading)**: Các tác vụ nặng như chạy AI nhận diện và tính toán căn chỉnh camera được đặt trong các luồng riêng biệt để tránh gây hiện tượng giật lag cho khung hình hiển thị.

- **Cơ chế Debounce (Lọc nhiễu)**: Trạng thái của một ô đỗ xe chỉ được cập nhật nếu kết quả nhận diện từ mô hình AI ổn định trong liên tiếp 5 khung hình.

## Lưu ý quan trọng

- **Reference Image**: Bắt buộc có tệp reference_frame.jpg tại thư mục gốc để làm mốc neo.

- **Video Source**: Mặc định là ./data/14191689_1920_1080_30fps.mp4. Bạn có thể tùy chỉnh đường dẫn trong code.
