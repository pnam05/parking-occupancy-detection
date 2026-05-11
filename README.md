# Smart Parking Management System

Hệ thống quản lý bãi đỗ xe thông minh sử dụng thị giác máy tính (**OpenCV**) và học sâu (**Deep Learning - MobileNetV3**). Hệ thống có khả năng tự động nhận diện trạng thái còn trống hoặc đã có xe của từng ô đỗ dựa trên vùng nhận diện (ROI) được thiết lập trước.

## Demo Hệ Thống

## [![Watch Demo](https://img.youtube.com/vi/uVK4E_3F79Q/0.jpg)](https://www.youtube.com/watch?v=uVK4E_3F79Q)

## Các tính năng nổi bật

- **ROI Setup Tool**: Công cụ thiết lập vùng đỗ xe linh hoạt hỗ trợ phóng to/thu nhỏ (Zoom) và di chuyển góc nhìn (Pan) để xử lý camera ở vị trí xa hoặc có độ phân giải cao.
- **Data Annotation**: Quy trình thu thập và gán nhãn dữ liệu trực tiếp từ nguồn video, cho phép tạo bộ dữ liệu huấn luyện nhanh chóng.
- **Optical Flow**: Sử dụng **Lucas-Kanade Optical Flow** kết hợp ma trận **Affine**, giúp bám sát độ rung lắc của camera với thời gian tính toán cực ngắn (< 2ms/frame).
- **Lightweight Model**: Sử dụng kiến trúc **MobileNetV3-Small** giúp tối ưu tốc độ xử lý (inference) và tiết kiệm tài nguyên phần cứng.

---

## Cấu trúc thư mục

- **data/14191689_1920_1080_30fps.mp4**: Video đầu vào.
- **weights/best.pth**: Trọng số mô hình AI đã huấn luyện.
- **draw_roi.py**: Công cụ xác định vị trí các ô đỗ xe trên khung hình video.
- **rois.json**: File lưu trữ tọa độ các ô đỗ xe.
- **collect_data.py**: Script cắt ảnh từ video và gán nhãn dữ liệu để chuẩn bị cho việc huấn luyện.
- **train_classification.py**: Thực hiện huấn luyện mô hình phân loại trạng thái ô đỗ xe.
- **config.py**: File cấu hình trung tâm (Đường dẫn, thông số AI).
- **stabilizer.py**: Module xử lý chống rung camera (VideoStabilizer).
- **model.py**: Module nạp AI Model (MobileNetV3) & Transforms.
- **utils.py**: Chứa các hàm hỗ trợ.
- **main.py**: Chương trình chính thực hiện nhận diện và hiển thị kết quả thời gian thực.

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

## Cơ chế tối ưu hiệu năng

- **Khởi động nóng (Warm-up)**: Ngay tại khung hình đầu tiên (Frame 0), hệ thống sẽ gửi một Batch lớn chứa toàn bộ các ô đỗ xe vào GPU để AI quét trạng thái 100% bãi đỗ, loại bỏ hoàn toàn độ trễ hiển thị lúc mới khởi động.

- **Cắt lát thời gian (Time-Slicing)**: Thay vì ép GPU nhận diện toàn bộ bãi xe ở mỗi frame, hệ thống cấu hình SLOTS_PER_FRAME để chỉ nhận diện luân phiên một vài ô đỗ (VD: 1 hoặc 5 ô) ở mỗi khung hình. Giúp chia nhỏ khối lượng công việc một cách hoàn hảo.

- **Bỏ qua khung hình (Frame-Skipping)**:Các tác vụ tính toán nặng (Optical Flow, AI) chỉ chạy ở các khung hình lẻ (1, 3, 5...). Các khung hình chẵn chỉ lấy kết quả tính toán trước đó để vẽ đồ họa (Render), giúp FPS tăng vọt.

- **Lọc nhiễu độ trễ thấp (Low Debounce)**: Nhờ khoảng trễ tự nhiên sinh ra từ thuật toán Time-Slicing, biến DEBOUNCE_THRESHOLD được hạ xuống thấp (2 khung hình) nhưng vẫn đảm bảo chống nhiễu sai lệch tuyệt đối.

## Lưu ý quan trọng

- **Video Source**: Mặc định là ./data/14191689_1920_1080_30fps.mp4. Bạn có thể tùy chỉnh đường dẫn trong config.py.
