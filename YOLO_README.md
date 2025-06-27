# 🚀 Hướng dẫn sử dụng YOLOv8 trong Fruit Recognition System

## 📋 Tổng quan

Hệ thống nhận diện quả thông minh đã được tích hợp đầy đủ YOLOv8 với các tính năng:

- **Upload ảnh**: Nhận diện và đếm quả từ ảnh tải lên
- **Real-time Detection**: Nhận diện real-time qua webcam
- **Batch Processing**: Xử lý nhiều ảnh cùng lúc
- **Thống kê chi tiết**: Theo dõi kết quả detection

## 🎯 Các tính năng chính

### 1. **YOLOv8 Upload** (Tab YOLOv8)
- **Chức năng**: Nhận diện và đếm quả/rau củ từ ảnh tải lên
- **Cách sử dụng**:
  1. Chuyển đến tab "YOLOv8"
  2. Chọn ảnh bằng cách kéo thả hoặc click "Chọn ảnh"
  3. Nhấn "Nhận diện & Đếm" để xem kết quả chi tiết
  4. Hoặc nhấn "Chỉ đếm" để chỉ xem số lượng

### 2. **YOLO Real-time** (Tab YOLO Real-time) ⭐ **MỚI**
- **Chức năng**: Nhận diện real-time qua webcam với YOLOv8
- **Cách sử dụng**:
  1. Chuyển đến tab "YOLO Real-time"
  2. Nhấn "Bật YOLO Webcam" để khởi động
  3. Hướng camera vào quả/rau củ cần nhận diện
  4. Xem kết quả real-time:
     - Tổng số quả phát hiện
     - Số loại quả khác nhau
     - FPS (tốc độ xử lý)
     - Chi tiết từng loại quả
  5. Các tùy chọn:
     - **Bật/tắt overlay**: Hiển thị/ẩn khung nhận diện
     - **Chụp ảnh**: Lưu ảnh hiện tại với kết quả detection

### 3. **Batch Processing**
- **Chức năng**: Xử lý nhiều ảnh cùng lúc
- **Cách sử dụng**:
  1. Chọn nhiều ảnh cùng lúc
  2. Hệ thống sẽ xử lý và hiển thị tổng kết

## 🔧 Cấu hình và tối ưu hóa

### Tham số detection:
- **Confidence Threshold**: 0.25 (có thể điều chỉnh)
- **IoU Threshold**: 0.45 (có thể điều chỉnh)
- **Real-time**: 0.3 confidence, 0.5 IoU (tối ưu tốc độ)

### Hiệu suất:
- **Upload**: Độ chính xác cao, xử lý đầy đủ
- **Real-time**: Tối ưu tốc độ, 10 FPS
- **Batch**: Xử lý song song, hiệu quả cao

## 📊 Kết quả detection

### Thông tin hiển thị:
- **Tổng số quả/rau củ** phát hiện
- **Chi tiết từng loại** với số lượng
- **Độ tin cậy** của mỗi detection
- **Ảnh có annotation** (bounding boxes)

### Loại quả được hỗ trợ:
- Apple, Banana, Orange, Tomato
- Carrot, Cucumber, Potato, Onion
- Garlic, Pepper, Lettuce, Cabbage
- Cauliflower, Broccoli, Corn, Peas
- Và nhiều loại khác...

## 🎮 Điều khiển

### Phím tắt:
- **Q**: Thoát (trong chế độ standalone)
- **Space**: Chụp ảnh (real-time)
- **O**: Bật/tắt overlay

### Giao diện:
- **Dark/Light mode**: Chuyển đổi theme
- **Responsive**: Tương thích mọi thiết bị
- **Real-time stats**: Theo dõi hiệu suất

## 🚀 Chạy ứng dụng

### 1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng web:
```bash
python app.py
```

### 3. Chạy standalone YOLO:
```bash
python detect_yolo.py
```

## 📁 Cấu trúc file

```
fruit_reconige_2.0/
├── app.py                 # Ứng dụng web chính
├── detect_yolo.py         # Module YOLO real-time
├── yolov8n.pt            # Model YOLOv8
├── templates/
│   └── index.html        # Giao diện web
└── uploads/              # Thư mục lưu ảnh
```

## 🔍 Troubleshooting

### Lỗi thường gặp:
1. **Webcam không hoạt động**:
   - Kiểm tra quyền truy cập camera
   - Thử refresh trang

2. **Detection chậm**:
   - Giảm độ phân giải ảnh
   - Tăng confidence threshold

3. **Model không load**:
   - Kiểm tra file `yolov8n.pt`
   - Cài đặt lại ultralytics

## 📈 Cải tiến tương lai

- [ ] Thêm model YOLOv8 custom cho quả Việt Nam
- [ ] Tối ưu hóa cho mobile
- [ ] Thêm tính năng tracking
- [ ] Export kết quả dạng video
- [ ] Tích hợp với database

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

---

**Lưu ý**: Đảm bảo có đủ RAM (4GB+) và GPU (khuyến nghị) để chạy YOLOv8 hiệu quả. 