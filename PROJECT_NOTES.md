# 📝 PROJECT NOTES - FRUIT RECOGNITION 2.0

## 🎯 TỔNG QUAN DỰ ÁN

**Tên dự án**: Fruit Recognition 2.0 - Hệ thống nhận diện trái cây thông minh  
**Mục tiêu**: Xây dựng ứng dụng web nhận diện 36 loại quả/rau củ bằng AI  
**Công nghệ chính**: CNN (MobileNetV2) + YOLO (YOLOv8) + Flask  

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

### Backend (Python/Flask)
- **Framework**: Flask
- **AI Models**: 
  - CNN: `fruit_classifier_mobilenetv2.h5` (MobileNetV2)
  - YOLO: `yolov8n.pt` (YOLOv8 nano)
- **Database**: JSON files (`users.json`, `detection_history.json`)
- **Authentication**: Session-based login

### Frontend (HTML/CSS/JavaScript)
- **Giao diện**: Responsive design
- **Tabs**: Upload Ảnh, Webcam, Lịch sử
- **Real-time**: Webcam processing
- **Visualization**: Bounding boxes, charts

---

## 📁 CẤU TRÚC THƯ MỤC

```
fruit_reconige_2.0/
├── app.py                 # Main Flask application
├── train_model.py         # CNN training script
├── requirements.txt       # Python dependencies
├── yolov8n.pt            # YOLO model file
├── templates/            # HTML templates
├── train/               # Training images (36 classes)
├── validation/          # Validation images
├── test/               # Test images
├── uploads/            # Uploaded images
└── users.json          # User database
```

---

## 🤖 AI MODELS

### 1. CNN Model (MobileNetV2)

#### **1.1. Định nghĩa CNN (Convolutional Neural Network)**
- **CNN** là kiến trúc mạng nơ-ron chuyên dụng cho việc xử lý dữ liệu có cấu trúc lưới như hình ảnh
- **Nguyên lý hoạt động**: Sử dụng các lớp tích chập (convolutional layers) để trích xuất đặc trưng từ hình ảnh
- **Ưu điểm**: Tự động học features, giảm tham số, hiệu quả với dữ liệu hình ảnh

#### **1.2. Các thành phần chính của CNN**
- **Convolutional Layer**: Lọc và trích xuất đặc trưng từ ảnh
- **Pooling Layer**: Giảm kích thước dữ liệu, tăng tính bất biến
- **Fully Connected Layer**: Phân loại cuối cùng
- **Activation Functions**: Kích hoạt phi tuyến (ReLU, Softmax)

#### **1.3. Kiến trúc MobileNetV2 trong dự án**
- **Mục đích**: Phân loại loại quả chính trong ảnh
- **Input**: 224x224x3 pixels (RGB)
- **Output**: 36 classes với softmax activation
- **Accuracy**: ~85-90%
- **Sử dụng**: Upload ảnh, Webcam capture
- **Transfer Learning**: Sử dụng pre-trained weights từ ImageNet

#### **1.4. Quá trình xử lý CNN**
```
Input Image (224x224x3) 
    ↓
MobileNetV2 Base (Feature Extraction)
    ↓
Global Average Pooling (Reduce dimensions)
    ↓
Dropout (0.3) - Regularization
    ↓
Dense Layer (36 classes)
    ↓
Softmax Activation
    ↓
Output: Class probabilities
```

### 2. YOLO Model (YOLOv8)

#### **2.1. Định nghĩa YOLO (You Only Look Once)**
- **YOLO** là thuật toán phát hiện đối tượng real-time
- **Nguyên lý**: Chia ảnh thành grid và dự đoán bounding boxes + class cho mỗi cell
- **Ưu điểm**: Tốc độ nhanh, có thể phát hiện nhiều đối tượng cùng lúc

#### **2.2. YOLOv8 trong dự án**
- **Mục đích**: Phát hiện và đếm nhiều quả trong ảnh
- **Input**: Any size image (tự động resize)
- **Output**: Bounding boxes + class labels + confidence scores
- **Speed**: Real-time (2 FPS)
- **Sử dụng**: Webcam scan mode
- **Confidence threshold**: 0.25
- **IoU threshold**: 0.45

### 3. Định nghĩa Train (Huấn luyện)

#### **3.1. Khái niệm Training**
- **Train** là quá trình dạy mô hình học từ dữ liệu để có thể dự đoán chính xác
- **Mục tiêu**: Tối ưu hóa các tham số của mô hình để giảm thiểu loss function
- **Kết quả**: Mô hình có khả năng tổng quát hóa tốt trên dữ liệu mới

#### **3.2. Quá trình Training trong dự án**
```
1. Chuẩn bị dữ liệu
   ├── Train set (70%): Huấn luyện mô hình
   ├── Validation set (15%): Đánh giá trong quá trình train
   └── Test set (15%): Đánh giá cuối cùng

2. Data Augmentation
   ├── Rotation (±30°)
   ├── Width/Height shift (±10%)
   ├── Shear (±20%)
   ├── Zoom (±20%)
   └── Horizontal flip

3. Model Architecture
   ├── Load MobileNetV2 pre-trained
   ├── Freeze base layers
   ├── Add custom layers
   └── Compile với optimizer

4. Training Process
   ├── Epochs: 40 (max)
   ├── Batch size: 32
   ├── Learning rate: 1e-4
   ├── Callbacks: Checkpoint, EarlyStopping
   └── Monitor: validation accuracy

5. Evaluation
   ├── Test accuracy
   ├── Confusion matrix
   └── Performance metrics
```

#### **3.3. Transfer Learning Strategy**
- **Base Model**: MobileNetV2 pre-trained trên ImageNet
- **Freeze Strategy**: Đóng băng tất cả layers của base model
- **Custom Layers**: Thêm GlobalAveragePooling2D + Dropout + Dense
- **Fine-tuning**: Chỉ train các layer mới thêm vào
- **Lợi ích**: Tận dụng features đã học, giảm thời gian train, tăng accuracy

#### **3.4. Training Parameters**
- **Optimizer**: Adam (learning_rate=1e-4)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**:
  - ModelCheckpoint: Lưu model tốt nhất
  - EarlyStopping: Dừng sớm nếu không cải thiện (patience=7)
- **Data Preprocessing**:
  - Resize: 224x224
  - Normalize: /255.0
  - Augmentation: Chỉ áp dụng cho train set

#### **3.5. Giải thích chi tiết code train_model.py**

##### **Import và Setup (Dòng 1-11)**
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
```
**Giải thích từng import:**
- **tensorflow**: Framework chính cho deep learning
- **image_dataset_from_directory**: Đọc ảnh từ thư mục tự động
- **MobileNetV2**: Mô hình CNN pre-trained hiệu quả
- **Dense, GlobalAveragePooling2D, Dropout**: Các layer neural network
- **Model**: Tạo mô hình tùy chỉnh
- **matplotlib**: Vẽ biểu đồ kết quả
- **ImageDataGenerator**: Tăng cường dữ liệu (data augmentation)
- **Adam**: Optimizer hiệu quả
- **ModelCheckpoint, EarlyStopping**: Callbacks để kiểm soát training

##### **Cấu hình tham số (Dòng 13-21)**
```python
train_dir = 'train'
val_dir = 'validation'
test_dir = 'test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
NUM_CLASSES = len(os.listdir('train'))
```
**Giải thích:**
- **train_dir, val_dir, test_dir**: Đường dẫn đến dữ liệu
- **IMG_SIZE**: Kích thước ảnh đầu vào (224x224 là chuẩn cho MobileNetV2)
- **BATCH_SIZE**: Số ảnh xử lý cùng lúc (32 là giá trị cân bằng)
- **EPOCHS**: Số lần lặp qua toàn bộ dữ liệu train
- **NUM_CLASSES**: Số loại quả (tự động đếm từ thư mục)

##### **Đọc dữ liệu (Dòng 23-41)**
```python
train_ds = image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
```
**Giải thích:**
- **label_mode='categorical'**: Mã hóa one-hot cho labels
- **image_size=IMG_SIZE**: Resize ảnh về 224x224
- **batch_size=BATCH_SIZE**: Chia dữ liệu thành batches
- **shuffle=True**: Xáo trộn dữ liệu train để tránh overfitting

##### **Data Augmentation (Dòng 51-65)**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
**Giải thích từng augmentation:**
- **rescale=1./255**: Chuẩn hóa pixel values về [0,1]
- **rotation_range=30**: Xoay ảnh ngẫu nhiên ±30°
- **width_shift_range=0.1**: Dịch chuyển ngang ±10%
- **height_shift_range=0.1**: Dịch chuyển dọc ±10%
- **shear_range=0.2**: Biến dạng cắt ±20%
- **zoom_range=0.2**: Phóng to/thu nhỏ ±20%
- **horizontal_flip=True**: Lật ngang ảnh
- **fill_mode='nearest'**: Cách điền pixel khi biến đổi

##### **Kiến trúc Model (Dòng 84-92)**
```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
```
**Giải thích từng layer:**
- **MobileNetV2**: Load pre-trained model từ ImageNet
- **include_top=False**: Không bao gồm classification layer cuối
- **base_model.trainable = False**: Đóng băng weights của base model
- **GlobalAveragePooling2D**: Giảm kích thước feature map
- **Dropout(0.3)**: Regularization để tránh overfitting
- **Dense(NUM_CLASSES, activation='softmax')**: Layer phân loại cuối
- **Model**: Tạo model hoàn chỉnh từ input đến output

##### **Compile và Callbacks (Dòng 94-99)**
```python
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('fruit_classifier_mobilenetv2.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
```
**Giải thích:**
- **Adam optimizer**: Optimizer hiệu quả với learning rate thấp
- **categorical_crossentropy**: Loss function cho multi-class classification
- **ModelCheckpoint**: Lưu model tốt nhất dựa trên validation accuracy
- **EarlyStopping**: Dừng train sớm nếu không cải thiện trong 7 epochs

##### **Training Process (Dòng 101-108)**
```python
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, earlystop]
)
```
**Giải thích:**
- **train_generator**: Dữ liệu train với augmentation
- **validation_data**: Dữ liệu validation để monitor
- **callbacks**: Các callback đã định nghĩa
- **history**: Lưu trữ metrics trong quá trình train

#### **3.6. Giải thích code CNN trong app.py**

##### **FruitClassifier Class (Dòng 351-397)**
```python
class FruitClassifier:
    def __init__(self, model_path='fruit_classifier_mobilenetv2.h5'):
        self.model = load_model(model_path)
        self.class_names = [
            'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
            'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
            'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
            'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
            'sweetpotato', 'tomato', 'turnip', 'watermelon'
        ]

    def preprocess_image(self, img):
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        preprocessed_img = self.preprocess_image(img)
        predictions = self.model.predict(preprocessed_img)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return predicted_class, confidence
```

**Giải thích từng method:**

###### **__init__ method (Dòng 352-358)**
- **load_model()**: Load mô hình đã train từ file .h5
- **class_names**: Danh sách tên 36 loại quả theo thứ tự
- **Lưu ý**: Thứ tự class_names phải khớp với thứ tự trong training

###### **preprocess_image method (Dòng 362-366)**
```python
def preprocess_image(self, img):
    img = cv2.resize(img, (224, 224))      # Resize về kích thước chuẩn
    img = img / 255.0                      # Normalize về [0,1]
    img = np.expand_dims(img, axis=0)      # Thêm batch dimension
    return img
```
**Giải thích từng bước:**
- **cv2.resize()**: Thay đổi kích thước ảnh về 224x224 (chuẩn MobileNetV2)
- **img / 255.0**: Chuẩn hóa pixel values từ [0,255] về [0,1]
- **np.expand_dims()**: Thêm dimension batch (từ (224,224,3) thành (1,224,224,3))

###### **predict method (Dòng 368-373)**
```python
def predict(self, img):
    preprocessed_img = self.preprocess_image(img)           # Tiền xử lý ảnh
    predictions = self.model.predict(preprocessed_img)      # Dự đoán
    predicted_class = self.class_names[np.argmax(predictions[0])]  # Lấy class có xác suất cao nhất
    confidence = float(np.max(predictions[0]))              # Lấy confidence score
    return predicted_class, confidence
```
**Giải thích từng bước:**
- **model.predict()**: Chạy inference trên mô hình CNN
- **np.argmax()**: Tìm index của class có xác suất cao nhất
- **class_names[index]**: Lấy tên class tương ứng
- **np.max()**: Lấy confidence score cao nhất

##### **Sử dụng CNN trong Upload (Dòng 465-530)**
```python
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Đọc và xử lý ảnh
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Dự đoán bằng CNN
        classifier = FruitClassifier()
        predicted_class, confidence = classifier.predict(img_rgb)
        
        # Lấy thông tin dinh dưỡng
        fruit_info = FRUIT_INFO.get(predicted_class, {})
        
        # Lưu lịch sử
        update_stats(predicted_class, confidence, img_base64, "upload", session.get('user_id'))
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'fruit_info': fruit_info
        })
```

**Giải thích quy trình:**
1. **Validate file**: Kiểm tra file upload hợp lệ
2. **Đọc ảnh**: Chuyển đổi file thành numpy array
3. **Chuyển màu**: BGR → RGB (OpenCV sử dụng BGR, CNN cần RGB)
4. **Dự đoán**: Sử dụng FruitClassifier để phân loại
5. **Lấy thông tin**: Tra cứu thông tin dinh dưỡng
6. **Lưu lịch sử**: Ghi lại kết quả detection
7. **Trả kết quả**: JSON response cho frontend

##### **Sử dụng CNN trong Webcam (Dòng 531-582)**
```python
@app.route('/webcam', methods=['POST'])
@login_required
def webcam_detection():
    # Nhận ảnh từ webcam (base64)
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    
    # Chuyển đổi thành numpy array
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Dự đoán bằng CNN
    classifier = FruitClassifier()
    predicted_class, confidence = classifier.predict(img_rgb)
    
    # Xử lý kết quả tương tự upload
```

**Giải thích:**
- **Base64 decode**: Chuyển đổi ảnh từ base64 string về bytes
- **cv2.imdecode()**: Chuyển bytes thành numpy array
- **Quy trình tương tự upload**: Sử dụng cùng CNN classifier

#### **3.7. So sánh CNN và YOLO trong dự án**

| Tiêu chí | CNN (MobileNetV2) | YOLO (YOLOv8) |
|----------|-------------------|---------------|
| **Mục đích** | Phân loại loại quả chính | Phát hiện nhiều quả cùng lúc |
| **Input** | 224x224x3 (cố định) | Any size (tự động resize) |
| **Output** | Class + Confidence | Bounding boxes + Classes + Confidence |
| **Tốc độ** | ~100ms per image | ~50ms per image |
| **Độ chính xác** | 85-90% | 70-80% |
| **Sử dụng** | Upload ảnh, Webcam capture | Webcam scan mode |
| **Ưu điểm** | Độ chính xác cao, đơn giản | Real-time, phát hiện nhiều đối tượng |
| **Nhược điểm** | Chỉ phân loại 1 quả chính | Độ chính xác thấp hơn |

#### **3.8. Tóm tắt quy trình hoàn chỉnh**

##### **Training Phase:**
1. **Data Preparation**: Chia dữ liệu thành train/validation/test
2. **Data Augmentation**: Tăng cường dữ liệu train
3. **Model Architecture**: MobileNetV2 + custom layers
4. **Transfer Learning**: Sử dụng pre-trained weights
5. **Training**: Adam optimizer, categorical crossentropy
6. **Evaluation**: Test accuracy, confusion matrix

##### **Inference Phase:**
1. **Image Preprocessing**: Resize, normalize, expand dimensions
2. **Model Prediction**: Forward pass qua CNN
3. **Post-processing**: Argmax để lấy class, max để lấy confidence
4. **Result Display**: Hiển thị kết quả + thông tin dinh dưỡng
5. **History Logging**: Lưu kết quả vào database

##### **Key Technical Points:**
- **Transfer Learning**: Tận dụng MobileNetV2 pre-trained
- **Data Augmentation**: Tăng robustness của model
- **Regularization**: Dropout để tránh overfitting
- **Early Stopping**: Tránh overfitting, tối ưu training time
- **Model Checkpointing**: Lưu model tốt nhất
- **Batch Processing**: Xử lý nhiều ảnh cùng lúc
- **Memory Optimization**: Prefetch để tăng hiệu suất

---

## 🔧 CÁC CHỨC NĂNG CHÍNH

### 1. Hệ thống Authentication
- ✅ Đăng ký tài khoản
- ✅ Đăng nhập/Đăng xuất
- ✅ Quên mật khẩu
- ✅ Session management

### 2. Nhận diện qua Upload
- ✅ Upload file ảnh (JPG, PNG)
- ✅ CNN classification
- ✅ Hiển thị kết quả + confidence
- ✅ Thông tin dinh dưỡng

### 3. Nhận diện qua Webcam
- ✅ Truy cập camera
- ✅ Real-time YOLO detection
- ✅ Bounding boxes visualization
- ✅ Đếm số lượng quả

### 4. Quản lý lịch sử
- ✅ Lưu tất cả detections
- ✅ Thống kê theo ngày/loại quả
- ✅ Export Excel
- ✅ Xóa lịch sử

---

## 📋 ĐỊNH NGHĨA CHI TIẾT CÁC CHỨC NĂNG

### 🔐 **1. HỆ THỐNG AUTHENTICATION**

#### **1.1. Đăng ký tài khoản (Register)**
- **Định nghĩa**: Cho phép người dùng tạo tài khoản mới
- **Chức năng**:
  - Nhập username và password
  - Validate thông tin (username không trùng, password đủ mạnh)
  - Mã hóa password bằng SHA-256
  - Lưu thông tin vào `users.json`
- **Bảo mật**: Password được hash trước khi lưu

#### **1.2. Đăng nhập (Login)**
- **Định nghĩa**: Xác thực người dùng để truy cập hệ thống
- **Chức năng**:
  - Kiểm tra username/password
  - Tạo session cho người dùng
  - Redirect đến trang chủ sau khi đăng nhập thành công
- **Session**: Lưu trạng thái đăng nhập trong 24 giờ

#### **1.3. Đăng xuất (Logout)**
- **Định nghĩa**: Kết thúc phiên làm việc của người dùng
- **Chức năng**: Xóa session và redirect về trang đăng nhập

#### **1.4. Quên mật khẩu (Forgot Password)**
- **Định nghĩa**: Khôi phục mật khẩu khi người dùng quên
- **Chức năng**: Reset password về mặc định (cần cải thiện)

### 📤 **2. NHẬN DIỆN QUA UPLOAD**

#### **2.1. Upload File**
- **Định nghĩa**: Tải lên ảnh từ thiết bị để phân tích
- **Chức năng**:
  - Hỗ trợ format JPG, PNG
  - Giới hạn kích thước 16MB
  - Validate file trước khi xử lý
  - Lưu ảnh vào thư mục `uploads/`

#### **2.2. CNN Classification**
- **Định nghĩa**: Phân loại loại quả chính trong ảnh bằng CNN
- **Chức năng**:
  - Tiền xử lý ảnh (resize 224x224, normalize)
  - Dự đoán bằng model MobileNetV2
  - Trả về tên quả và độ tin cậy (confidence)
- **Độ chính xác**: 85-90%

#### **2.3. Hiển thị kết quả**
- **Định nghĩa**: Trình bày kết quả nhận diện cho người dùng
- **Chức năng**:
  - Hiển thị ảnh gốc
  - Tên quả (tiếng Anh và tiếng Việt)
  - Độ tin cậy dưới dạng phần trăm
  - Thông tin dinh dưỡng chi tiết

### 📹 **3. NHẬN DIỆN QUA WEBCAM**

#### **3.1. Truy cập Camera**
- **Định nghĩa**: Kết nối với camera của thiết bị
- **Chức năng**:
  - Yêu cầu quyền truy cập camera
  - Hiển thị video stream real-time
  - Hỗ trợ camera trước/sau
- **Độ phân giải**: 640x480 (tối ưu)

#### **3.2. Chế độ Scan (YOLO)**
- **Định nghĩa**: Phát hiện liên tục nhiều quả trong khung hình
- **Chức năng**:
  - Tự động scan mỗi 500ms (2 FPS)
  - Phát hiện bounding boxes cho từng quả
  - Đếm số lượng mỗi loại quả
  - Hiển thị overlay thông tin
- **Model**: YOLOv8 với confidence threshold 0.25

#### **3.3. Chế độ Capture (CNN)**
- **Định nghĩa**: Chụp ảnh và phân tích bằng CNN
- **Chức năng**:
  - Chụp ảnh từ webcam
  - Phân loại loại quả chính
  - Hiển thị kết quả chi tiết
- **Model**: MobileNetV2

#### **3.4. Bounding Boxes Visualization**
- **Định nghĩa**: Vẽ khung và nhãn cho các quả được phát hiện
- **Chức năng**:
  - Vẽ rectangle xung quanh quả
  - Hiển thị tên quả và confidence
  - Màu sắc khác nhau cho từng loại quả

### 📊 **4. QUẢN LÝ LỊCH SỬ**

#### **4.1. Lưu trữ Detection History**
- **Định nghĩa**: Ghi lại tất cả các lần nhận diện
- **Chức năng**:
  - Lưu thông tin: thời gian, loại quả, confidence
  - Lưu ảnh dưới dạng base64
  - Phân loại theo loại detection (upload/webcam/yolo)
  - Liên kết với user ID

#### **4.2. Thống kê tổng quan**
- **Định nghĩa**: Hiển thị số liệu tổng hợp về hoạt động
- **Chức năng**:
  - Tổng số lần nhận diện
  - Số loại quả khác nhau đã phát hiện
  - Biểu đồ theo thời gian
  - Top loại quả được nhận diện nhiều nhất

#### **4.3. Thống kê chi tiết**
- **Định nghĩa**: Phân tích sâu về dữ liệu nhận diện
- **Chức năng**:
  - Thống kê theo ngày/tuần/tháng
  - Phân tích theo loại quả
  - Độ tin cậy trung bình
  - So sánh hiệu suất các model

#### **4.4. Export Excel**
- **Định nghĩa**: Xuất dữ liệu lịch sử ra file Excel
- **Chức năng**:
  - Tạo file .xlsx với đầy đủ thông tin
  - Bao gồm: thời gian, loại quả, confidence, ảnh
  - Hỗ trợ filter và sort
- **Format**: Pandas DataFrame → Excel

#### **4.5. Xóa lịch sử**
- **Định nghĩa**: Xóa các bản ghi lịch sử
- **Chức năng**:
  - Xóa từng bản ghi riêng lẻ
  - Xóa tất cả lịch sử
  - Xác nhận trước khi xóa
- **Bảo mật**: Chỉ user sở hữu mới được xóa

### 🍎 **5. THÔNG TIN DINH DƯỠNG**

#### **5.1. Database thông tin quả**
- **Định nghĩa**: Cơ sở dữ liệu thông tin chi tiết về từng loại quả
- **Chức năng**:
  - Tên tiếng Việt
  - Calories/100g
  - Danh sách vitamin
  - Lợi ích sức khỏe
  - Màu sắc hiển thị

#### **5.2. Hiển thị thông tin**
- **Định nghĩa**: Trình bày thông tin dinh dưỡng cho người dùng
- **Chức năng**:
  - Card thông tin đẹp mắt
  - Tags vitamin với màu sắc
  - Mô tả lợi ích sức khỏe
  - Responsive design

### 🔧 **6. CÁC CHỨC NĂNG PHỤ TRỢ**

#### **6.1. Session Management**
- **Định nghĩa**: Quản lý phiên làm việc của người dùng
- **Chức năng**:
  - Tạo session khi đăng nhập
  - Kiểm tra session cho các trang bảo vệ
  - Tự động logout sau thời gian timeout
- **Bảo mật**: Secret key cho session

#### **6.2. File Management**
- **Định nghĩa**: Quản lý file upload và lưu trữ
- **Chức năng**:
  - Validate file type và size
  - Tạo tên file unique
  - Lưu trữ có tổ chức
  - Cleanup file cũ

#### **6.3. Error Handling**
- **Định nghĩa**: Xử lý lỗi và thông báo cho người dùng
- **Chức năng**:
  - Catch và log lỗi
  - Hiển thị thông báo thân thiện
  - Fallback khi model lỗi
  - Debug information

#### **6.4. Responsive Design**
- **Định nghĩa**: Giao diện thích ứng với mọi thiết bị
- **Chức năng**:
  - Mobile-first design
  - Flexible layout
  - Touch-friendly interface
  - Cross-browser compatibility

---

## 📊 DỮ LIỆU VÀ THÔNG TIN

### 36 Loại Quả/Rau Củ
1. apple, banana, beetroot, bell pepper
2. cabbage, capsicum, carrot, cauliflower
3. chilli pepper, corn, cucumber, eggplant
4. garlic, ginger, grapes, jalepeno
5. kiwi, lemon, lettuce, mango
6. onion, orange, paprika, pear
7. peas, pineapple, pomegranate, potato
8. raddish, soy beans, spinach, sweetcorn
9. sweetpotato, tomato, turnip, watermelon

### Thông tin dinh dưỡng cho mỗi loại:
- Tên tiếng Việt
- Calories/100g
- Danh sách vitamin
- Lợi ích sức khỏe
- Màu sắc hiển thị

---

## 🚀 CÁCH CHẠY DỰ ÁN

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
- Đặt ảnh training vào thư mục `train/`
- Đặt ảnh validation vào thư mục `validation/`
- Đặt ảnh test vào thư mục `test/`

### 3. Train CNN model (nếu cần)
```bash
python train_model.py
```

### 4. Chạy ứng dụng
```bash
python app.py
# hoặc
python run_webapp.py
```

### 5. Truy cập
- URL: http://localhost:5000
- Đăng ký tài khoản mới hoặc đăng nhập

---

## 🔍 API ENDPOINTS

### Authentication
- `GET /` - Trang chủ (yêu cầu login)
- `GET /login` - Trang đăng nhập
- `POST /login` - Xử lý đăng nhập
- `GET /logout` - Đăng xuất
- `GET /register` - Trang đăng ký
- `POST /register` - Xử lý đăng ký

### Detection
- `POST /upload` - Upload ảnh (CNN)
- `POST /webcam` - Webcam capture (CNN)
- `POST /yolo-detect` - YOLO detection

### History & Stats
- `GET /history` - Lịch sử detections
- `GET /stats` - Thống kê tổng quan
- `GET /history-stats` - Thống kê chi tiết
- `POST /delete_history` - Xóa lịch sử
- `GET /export_history_excel` - Export Excel

---

## ⚙️ CẤU HÌNH QUAN TRỌNG

### Model Paths
- CNN Model: `fruit_classifier_mobilenetv2.h5`
- YOLO Model: `yolov8n.pt`

### Image Settings
- Max file size: 16MB
- Supported formats: JPG, PNG
- CNN input size: 224x224
- YOLO: Any size

### Webcam Settings
- Resolution: 640x480 (ideal)
- Scan interval: 500ms (2 FPS)
- Confidence threshold: 0.25 (YOLO)

---

## 🐛 TROUBLESHOOTING

### Lỗi thường gặp:
1. **Model không load**: Kiểm tra file model tồn tại
2. **Webcam không hoạt động**: Kiểm tra quyền truy cập camera
3. **Upload lỗi**: Kiểm tra định dạng file và kích thước
4. **Memory error**: Giảm batch size hoặc image size

### Debug tips:
- Kiểm tra console logs
- Xem Flask debug output
- Test từng component riêng lẻ

---

## 📈 METRICS VÀ ĐÁNH GIÁ

### Model Performance
- CNN Accuracy: ~85-90%
- YOLO mAP: ~70-80%
- Inference time: <1s per image

### User Experience
- Response time: <2s
- UI/UX: Intuitive
- Mobile friendly: Yes

---

## 🔮 PHÁT TRIỂN TƯƠNG LAI

### Tính năng có thể thêm:
1. **Mobile App**: React Native/Flutter
2. **Cloud Deployment**: AWS/Azure
3. **Database**: PostgreSQL/MongoDB
4. **Real-time API**: WebSocket
5. **Multi-language**: English, Chinese
6. **Advanced Analytics**: ML insights
7. **Barcode Integration**: Product lookup
8. **Social Features**: Share results

### Cải thiện AI:
1. **More classes**: 100+ fruit types
2. **Better accuracy**: Ensemble models
3. **Faster inference**: Model optimization
4. **Edge deployment**: TensorFlow Lite

---

## 📚 TÀI LIỆU THAM KHẢO

### Technologies
- Flask Documentation
- TensorFlow/Keras Guides
- YOLOv8 Documentation
- OpenCV Tutorials

### Research Papers
- MobileNetV2: Inverted Residuals and Linear Bottlenecks
- YOLOv8: You Only Look Once v8
- Transfer Learning for Image Classification

---

## 👥 TEAM & CONTRIBUTION

**Developer**: [Your Name]  
**Version**: 2.0  
**Last Updated**: [Date]  
**License**: MIT  

---



