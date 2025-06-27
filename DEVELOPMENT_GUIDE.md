# 🚀 Hướng dẫn phát triển hệ thống nhận diện trái cây

## 📋 Tổng quan dự án

Hệ thống nhận diện trái cây sử dụng AI để nhận diện 36 loại quả/rau củ từ hình ảnh, cung cấp thông tin dinh dưỡng và lưu trữ lịch sử nhận diện.

---

## 🎯 Bước 1: Chuẩn bị môi trường

### 1.1 Cài đặt Python và dependencies
```bash
# Cài đặt Python 3.7+
python --version

# Tạo virtual environment
python -m venv fruit_env
source fruit_env/bin/activate  # Linux/Mac
# hoặc
fruit_env\Scripts\activate     # Windows

# Cài đặt dependencies
pip install flask tensorflow opencv-python pillow ultralytics pandas openpyxl
```

### 1.2 Tạo cấu trúc thư mục
```
fruit_reconige_2.0/
├── app.py                          # Main application
├── templates/                      # HTML templates
│   ├── index.html                 # Main interface
│   ├── login.html                 # Login page
│   ├── register.html              # Register page
│   └── forgot_password.html       # Forgot password
├── uploads/                       # Uploaded images
├── train/                         # Training dataset
├── validation/                    # Validation dataset
├── test/                         # Test dataset
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🎯 Bước 2: Thu thập và chuẩn bị dữ liệu

### 2.1 Thu thập dataset
```bash
# Tạo thư mục cho từng loại quả
mkdir -p train/apple train/banana train/orange train/tomato
mkdir -p validation/apple validation/banana validation/orange validation/tomato
mkdir -p test/apple test/banana test/orange test/tomato

# Thu thập ảnh cho từng loại (ít nhất 50-100 ảnh/loại)
# Có thể sử dụng:
# - Kaggle datasets
# - Google Images
# - Flickr API
# - Tự chụp ảnh
```

### 2.2 Tiền xử lý dữ liệu
```python
# Script tiền xử lý ảnh
import cv2
import os
from PIL import Image

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    """Tiền xử lý ảnh cho training"""
    for fruit_dir in os.listdir(input_dir):
        fruit_path = os.path.join(input_dir, fruit_dir)
        if os.path.isdir(fruit_path):
            output_fruit_path = os.path.join(output_dir, fruit_dir)
            os.makedirs(output_fruit_path, exist_ok=True)
            
            for img_file in os.listdir(fruit_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(fruit_path, img_file)
                    
                    # Đọc và resize ảnh
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, target_size)
                    
                    # Lưu ảnh đã xử lý
                    output_path = os.path.join(output_fruit_path, img_file)
                    cv2.imwrite(output_path, img)

# Chạy tiền xử lý
preprocess_images('raw_data', 'train', (224, 224))
```

---

## 🎯 Bước 3: Huấn luyện mô hình AI

### 3.1 Tạo script huấn luyện
```python
# train_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import os

def create_model(num_classes):
    """Tạo model MobileNetV2 với transfer learning"""
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model():
    """Huấn luyện model"""
    # Cấu hình
    img_height, img_width = 224, 224
    batch_size = 32
    epochs = 50
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        'validation',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Tạo model
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Huấn luyện
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    # Lưu model
    model.save('fruit_classifier_mobilenetv2.h5')
    print("Model đã được lưu!")

if __name__ == "__main__":
    train_model()
```

### 3.2 Chạy huấn luyện
```bash
python train_model.py
```

---

## 🎯 Bước 4: Tạo Flask Web Application

### 4.1 Tạo file app.py cơ bản
```python
# app.py - Phiên bản cơ bản
from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'

# Tạo thư mục uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Danh sách các loại quả
CLASS_NAMES = ['apple', 'banana', 'orange', 'tomato']  # Thêm các loại khác

# Load model
try:
    model = load_model('fruit_classifier_mobilenetv2.h5')
    print("Model loaded successfully!")
except:
    print("Error loading model!")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Lưu file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Xử lý ảnh
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Dự đoán
        if model:
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            fruit_name = CLASS_NAMES[predicted_class]
            
            return jsonify({
                'success': True,
                'fruit_name': fruit_name,
                'confidence': float(confidence)
            })
        else:
            return jsonify({'error': 'Model not loaded'})

if __name__ == '__main__':
    app.run(debug=True)
```

### 4.2 Tạo template HTML cơ bản
```html
<!-- templates/index.html - Phiên bản cơ bản -->
<!DOCTYPE html>
<html>
<head>
    <title>Fruit Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; }
        .result { margin-top: 20px; padding: 10px; background: #f0f0f0; }
    </style>
</head>
<body>
    <h1>Fruit Recognition System</h1>
    
    <div class="upload-area">
        <input type="file" id="fileInput" accept="image/*">
        <button onclick="uploadImage()">Upload and Analyze</button>
    </div>
    
    <div class="result" id="result" style="display: none;">
        <h3>Result:</h3>
        <p id="fruitName"></p>
        <p id="confidence"></p>
    </div>

    <script>
        async function uploadImage() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('fruitName').textContent = 'Fruit: ' + result.fruit_name;
                    document.getElementById('confidence').textContent = 'Confidence: ' + (result.confidence * 100).toFixed(2) + '%';
                    document.getElementById('result').style.display = 'block';
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error uploading file');
            }
        }
    </script>
</body>
</html>
```

---

## 🎯 Bước 5: Thêm tính năng Authentication

### 5.1 Tạo hệ thống đăng nhập
```python
# Thêm vào app.py
from flask import session, redirect, url_for
import hashlib

# User management functions
def load_users():
    if not os.path.exists('users.json'):
        return {}
    with open('users.json', 'r') as f:
        return json.load(f)

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        users = load_users()
        if username in users and users[username]['password'] == hash_password(password):
            session['logged_in'] = True
            session['username'] = username
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Invalid credentials'})
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Bảo vệ route chính
@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))
```

### 5.2 Tạo template login
```html
<!-- templates/login.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Login - Fruit Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .login-form { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        input { display: block; margin: 10px 0; padding: 10px; width: 250px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="login-form">
        <h2>Login</h2>
        <input type="text" id="username" placeholder="Username">
        <input type="password" id="password" placeholder="Password">
        <button onclick="login()">Login</button>
        <p id="message"></p>
    </div>

    <script>
        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            const response = await fetch('/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password})
            });
            
            const result = await response.json();
            
            if (result.success) {
                window.location.href = '/';
            } else {
                document.getElementById('message').textContent = result.error;
            }
        }
    </script>
</body>
</html>
```

---

## 🎯 Bước 6: Thêm tính năng History

### 6.1 Tạo hệ thống lưu trữ lịch sử
```python
# Thêm vào app.py
# Global variables for history
detection_history = []

def save_history():
    with open('detection_history.json', 'w') as f:
        json.dump(detection_history, f, indent=2, default=str)

def load_history():
    global detection_history
    if os.path.exists('detection_history.json'):
        with open('detection_history.json', 'r') as f:
            detection_history = json.load(f)

# Load history khi khởi động
load_history()

# Cập nhật route upload để lưu history
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    # ... existing code ...
    
    if model:
        # ... prediction code ...
        
        # Lưu vào history
        history_item = {
            'id': f"det_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'fruit_name': fruit_name,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat(),
            'user': session.get('username'),
            'filename': file.filename
        }
        
        detection_history.append(history_item)
        save_history()
        
        return jsonify({
            'success': True,
            'fruit_name': fruit_name,
            'confidence': float(confidence)
        })

@app.route('/history')
@login_required
def get_history():
    return jsonify({'history': detection_history[-50:]})  # 50 items gần nhất
```

### 6.2 Thêm tab History vào frontend
```html
<!-- Thêm vào templates/index.html -->
<div class="tab-bar">
    <button class="tab-btn active" onclick="showTab('upload')">Upload</button>
    <button class="tab-btn" onclick="showTab('history')">History</button>
</div>

<div id="upload-tab" class="tab-content active">
    <!-- Existing upload content -->
</div>

<div id="history-tab" class="tab-content">
    <h3>Detection History</h3>
    <div id="historyList"></div>
</div>

<script>
// Thêm functions cho history
async function loadHistory() {
    const response = await fetch('/history');
    const data = await response.json();
    
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = data.history.map(item => `
        <div class="history-item">
            <strong>${item.fruit_name}</strong> - ${(item.confidence * 100).toFixed(2)}%
            <br><small>${new Date(item.timestamp).toLocaleString()}</small>
        </div>
    `).join('');
}

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
    
    if (tabName === 'history') {
        loadHistory();
    }
}
</script>
```

---

## 🎯 Bước 7: Thêm thông tin dinh dưỡng

### 7.1 Tạo database thông tin quả
```python
# Thêm vào app.py
FRUIT_INFO = {
    'apple': {
        'name_vi': 'Táo',
        'calories': 52,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Tốt cho tim mạch, chống oxy hóa'
    },
    'banana': {
        'name_vi': 'Chuối',
        'calories': 89,
        'vitamins': ['Vitamin B6', 'Vitamin C', 'Potassium'],
        'benefits': 'Tốt cho tiêu hóa, cung cấp năng lượng'
    },
    # Thêm thông tin cho các loại quả khác
}

@app.route('/fruit-info/<fruit_name>')
@login_required
def get_fruit_info(fruit_name):
    fruit_info = FRUIT_INFO.get(fruit_name, {})
    return jsonify({'success': True, 'info': fruit_info})
```

### 7.2 Cập nhật frontend để hiển thị thông tin
```javascript
// Thêm vào templates/index.html
async function displayFruitInfo(fruitName) {
    try {
        const response = await fetch(`/fruit-info/${fruitName}`);
        const data = await response.json();
        
        if (data.success) {
            const info = data.info;
            document.getElementById('fruitInfo').innerHTML = `
                <h4>Thông tin dinh dưỡng:</h4>
                <p><strong>Tên tiếng Việt:</strong> ${info.name_vi}</p>
                <p><strong>Calories:</strong> ${info.calories} kcal/100g</p>
                <p><strong>Vitamin:</strong> ${info.vitamins.join(', ')}</p>
                <p><strong>Lợi ích:</strong> ${info.benefits}</p>
            `;
            document.getElementById('fruitInfo').style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading fruit info:', error);
    }
}

// Cập nhật function upload để gọi displayFruitInfo
async function uploadImage() {
    // ... existing code ...
    
    if (result.success) {
        document.getElementById('fruitName').textContent = 'Fruit: ' + result.fruit_name;
        document.getElementById('confidence').textContent = 'Confidence: ' + (result.confidence * 100).toFixed(2) + '%';
        document.getElementById('result').style.display = 'block';
        
        // Hiển thị thông tin dinh dưỡng
        await displayFruitInfo(result.fruit_name);
    }
}
```

---

## 🎯 Bước 8: Thêm tính năng Webcam

### 8.1 Cài đặt YOLO cho real-time detection
```python
# Thêm vào app.py
from ultralytics import YOLO

# Load YOLO model
try:
    yolo_model = YOLO('yolov8n.pt')
    print("YOLO model loaded!")
except:
    print("Error loading YOLO model!")
    yolo_model = None

@app.route('/webcam-detect', methods=['POST'])
@login_required
def webcam_detection():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and yolo_model:
        # Lưu file tạm
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam.jpg')
        file.save(filename)
        
        # Detect với YOLO
        results = yolo_model(filename)
        
        # Xử lý kết quả
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
        
        return jsonify({
            'success': True,
            'detections': detections
        })
    
    return jsonify({'error': 'Model not loaded'})
```

### 8.2 Thêm webcam interface
```html
<!-- Thêm vào templates/index.html -->
<div id="webcam-tab" class="tab-content">
    <h3>Webcam Detection</h3>
    <video id="webcam" width="640" height="480" autoplay></video>
    <br>
    <button onclick="startWebcam()">Start Webcam</button>
    <button onclick="captureAndDetect()">Capture & Detect</button>
    <div id="webcamResult"></div>
</div>

<script>
let webcamStream = null;

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('webcam');
        video.srcObject = stream;
        webcamStream = stream;
    } catch (error) {
        alert('Error accessing webcam: ' + error.message);
    }
}

async function captureAndDetect() {
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'webcam.jpg');
        
        try {
            const response = await fetch('/webcam-detect', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                const detections = result.detections.map(d => 
                    `${d.class} (${(d.confidence * 100).toFixed(1)}%)`
                ).join(', ');
                
                document.getElementById('webcamResult').innerHTML = 
                    `<p>Detected: ${detections}</p>`;
            } else {
                document.getElementById('webcamResult').innerHTML = 
                    `<p>Error: ${result.error}</p>`;
            }
        } catch (error) {
            document.getElementById('webcamResult').innerHTML = 
                '<p>Error processing image</p>';
        }
    });
}
</script>
```

---

## 🎯 Bước 9: Cải thiện giao diện và UX

### 9.1 Thêm CSS styling
```css
/* Thêm vào templates/index.html */
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    text-align: center;
}

.tab-bar {
    display: flex;
    background: #f8f9fa;
    border-bottom: 1px solid #dee2e6;
}

.tab-btn {
    flex: 1;
    padding: 15px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s;
}

.tab-btn.active {
    background: #667eea;
    color: white;
}

.tab-content {
    display: none;
    padding: 30px;
}

.tab-content.active {
    display: block;
}

.upload-area {
    border: 3px dashed #667eea;
    border-radius: 10px;
    padding: 40px;
    text-align: center;
    margin-bottom: 20px;
    transition: all 0.3s;
}

.upload-area:hover {
    border-color: #764ba2;
    background: #f8f9fa;
}

.btn {
    background: #667eea;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s;
}

.btn:hover {
    background: #764ba2;
    transform: translateY(-2px);
}

.result {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}

.history-item {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    transition: all 0.3s;
}

.history-item:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
</style>
```

### 9.2 Thêm animations và loading states
```javascript
// Thêm vào templates/index.html
function showLoading(show) {
    const loadingDiv = document.getElementById('loading');
    if (show) {
        loadingDiv.innerHTML = '<div class="spinner"></div><p>Đang xử lý...</p>';
        loadingDiv.style.display = 'block';
    } else {
        loadingDiv.style.display = 'none';
    }
}

function showMessage(message, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
}

// Cập nhật upload function
async function uploadImage() {
    showLoading(true);
    
    try {
        // ... existing upload code ...
        
        if (result.success) {
            showMessage('Nhận diện thành công!', 'success');
        } else {
            showMessage('Lỗi: ' + result.error, 'error');
        }
    } catch (error) {
        showMessage('Lỗi kết nối!', 'error');
    } finally {
        showLoading(false);
    }
}
```

---

## 🎯 Bước 10: Testing và Deployment

### 10.1 Tạo test script
```python
# test_system.py
import requests
import json
import os

def test_system():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Fruit Recognition System...")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(base_url)
        print("✅ Server is running")
    except:
        print("❌ Server is not running")
        return
    
    # Test 2: Test login
    login_data = {
        'username': 'testuser',
        'password': 'testpass'
    }
    
    response = requests.post(f"{base_url}/login", json=login_data)
    if response.status_code == 200:
        print("✅ Login endpoint working")
    else:
        print("❌ Login endpoint error")
    
    # Test 3: Test upload (if you have a test image)
    if os.path.exists('test_image.jpg'):
        with open('test_image.jpg', 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ Upload and recognition working")
                else:
                    print("❌ Recognition failed")
            else:
                print("❌ Upload endpoint error")
    else:
        print("ℹ️  No test image found")
    
    print("🎯 Testing completed!")

if __name__ == "__main__":
    test_system()
```

### 10.2 Tạo requirements.txt
```txt
# requirements.txt
Flask==2.3.3
tensorflow==2.13.0
opencv-python==4.8.0.76
Pillow==10.0.0
ultralytics==8.0.196
pandas==2.0.3
openpyxl==3.1.2
numpy==1.24.3
Werkzeug==2.3.7
```

### 10.3 Tạo README.md
```markdown
# Fruit Recognition System

Hệ thống nhận diện trái cây sử dụng AI.

## Cài đặt

1. Clone repository
2. Cài đặt dependencies: `pip install -r requirements.txt`
3. Chạy: `python app.py`
4. Truy cập: `http://localhost:5000`

## Tính năng

- Nhận diện 36 loại quả/rau củ
- Upload ảnh và webcam detection
- Thông tin dinh dưỡng chi tiết
- Lịch sử nhận diện
- Hệ thống đăng nhập

## Cấu trúc

- `app.py`: Main application
- `templates/`: HTML templates
- `uploads/`: Uploaded images
- `*.h5`: AI models
- `*.json`: Data files
```

---

## 🎯 Bước 11: Production Deployment

### 11.1 Cấu hình production
```python
# production_config.py
import os

class ProductionConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-production-secret-key'
    DEBUG = False
    UPLOAD_FOLDER = '/var/www/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Database configuration (if using)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block'
    }
```

### 11.2 Tạo WSGI file
```python
# wsgi.py
from app import app

if __name__ == "__main__":
    app.run()
```

### 11.3 Tạo systemd service
```ini
# /etc/systemd/system/fruit-recognition.service
[Unit]
Description=Fruit Recognition Web App
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/fruit-recognition
Environment="PATH=/var/www/fruit-recognition/venv/bin"
ExecStart=/var/www/fruit-recognition/venv/bin/gunicorn --workers 3 --bind unix:fruit-recognition.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
```

---

## 📝 Checklist hoàn thành

### ✅ Bước 1-3: Foundation
- [ ] Môi trường Python setup
- [ ] Dataset thu thập và xử lý
- [ ] Model AI huấn luyện thành công

### ✅ Bước 4-6: Core Features
- [ ] Flask app cơ bản
- [ ] Upload và nhận diện
- [ ] Authentication system
- [ ] History management

### ✅ Bước 7-9: Advanced Features
- [ ] Thông tin dinh dưỡng
- [ ] Webcam detection
- [ ] UI/UX improvements

### ✅ Bước 10-11: Production
- [ ] Testing completed
- [ ] Documentation ready
- [ ] Production deployment

---

## 🚀 Tips và Best Practices

1. **Data Quality**: Đảm bảo dataset chất lượng cao
2. **Model Optimization**: Fine-tune model cho accuracy tốt nhất
3. **Security**: Implement proper authentication và validation
4. **Performance**: Optimize image processing và database queries
5. **User Experience**: Focus on intuitive UI và fast response times
6. **Monitoring**: Add logging và error tracking
7. **Backup**: Regular backup của data và models
8. **Scaling**: Plan for horizontal scaling nếu cần

