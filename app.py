from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, send_file
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import webbrowser
import threading
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from functools import wraps
import pandas as pd
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'fruit_secret_key_2024'  # Thêm secret key cho session

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Danh sách các loại quả
CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Màu sắc cho từng loại quả
FRUIT_COLORS = {
    'apple': '#FF0000',      # Đỏ
    'banana': '#FFFF00',     # Vàng
    'beetroot': '#FF00FF',   # Tím
    'bell pepper': '#FF00FF', # Tím
    'cabbage': '#FF00FF',    # Tím
    'capsicum': '#FF00FF',   # Tím
    'carrot': '#FFA500',     # Cam
    'cauliflower': '#FF00FF', # Tím
    'chilli pepper': '#FF00FF',# Tím
    'corn': '#FF00FF',       # Tím
    'cucumber': '#00FF00',   # Xanh lá
    'eggplant': '#FF00FF',   # Tím
    'garlic': '#FF00FF',     # Tím
    'ginger': '#FF00FF',     # Tím
    'grapes': '#800080',     # Tím
    'jalepeno': '#FF00FF',   # Tím
    'kiwi': '#90EE90',       # Xanh nhạt
    'lemon': '#FFFF00',      # Vàng
    'lettuce': '#FF00FF',    # Tím
    'mango': '#FF8C00',      # Cam đậm
    'onion': '#FF00FF',      # Tím
    'orange': '#FFA500',     # Cam
    'paprika': '#FF00FF',    # Tím
    'pear': '#FF00FF',       # Tím
    'peas': '#FF00FF',       # Tím
    'pineapple': '#FFD700',  # Vàng đậm
    'pomegranate': '#FF00FF',# Tím
    'potato': '#FF00FF',     # Tím
    'raddish': '#FF00FF',    # Tím
    'soy beans': '#FF00FF',   # Tím
    'spinach': '#FF00FF',    # Tím
    'sweetcorn': '#FF00FF',   # Tím
    'sweetpotato': '#FF00FF', # Tím
    'tomato': '#FF00FF',     # Tím
    'turnip': '#FF00FF',      # Tím
    'watermelon': '#FF69B4', # Hồng
    'default': '#FFFFFF'     # Trắng
}

# Thông tin dinh dưỡng cho từng loại quả
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
    'beetroot': {
        'name_vi': 'Củ dền',
        'calories': 43,
        'vitamins': ['Folate', 'Vitamin C'],
        'benefits': 'Tốt cho máu, hỗ trợ huyết áp'
    },
    'bell pepper': {
        'name_vi': 'Ớt chuông',
        'calories': 31,
        'vitamins': ['Vitamin C', 'Vitamin A', 'Vitamin B6'],
        'benefits': 'Tăng cường miễn dịch, tốt cho mắt'
    },
    'cabbage': {
        'name_vi': 'Bắp cải',
        'calories': 25,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Tốt cho tiêu hóa, chống viêm'
    },
    'capsicum': {
        'name_vi': 'Ớt',
        'calories': 40,
        'vitamins': ['Vitamin C', 'Vitamin A'],
        'benefits': 'Tăng cường trao đổi chất, chống oxy hóa'
    },
    'carrot': {
        'name_vi': 'Cà rốt',
        'calories': 41,
        'vitamins': ['Vitamin A', 'Vitamin K'],
        'benefits': 'Tốt cho mắt, chống lão hóa'
    },
    'cauliflower': {
        'name_vi': 'Súp lơ',
        'calories': 25,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Chống ung thư, tốt cho xương'
    },
    'chilli pepper': {
        'name_vi': 'Ớt cay',
        'calories': 40,
        'vitamins': ['Vitamin C', 'Vitamin A'],
        'benefits': 'Tăng cường trao đổi chất, hỗ trợ tiêu hóa'
    },
    'corn': {
        'name_vi': 'Ngô',
        'calories': 86,
        'vitamins': ['Vitamin B', 'Folate'],
        'benefits': 'Cung cấp năng lượng, tốt cho mắt'
    },
    'cucumber': {
        'name_vi': 'Dưa chuột',
        'calories': 16,
        'vitamins': ['Vitamin K', 'Vitamin C'],
        'benefits': 'Giải nhiệt, tốt cho da'
    },
    'eggplant': {
        'name_vi': 'Cà tím',
        'calories': 25,
        'vitamins': ['Vitamin B6', 'Vitamin K'],
        'benefits': 'Chống oxy hóa, tốt cho tim mạch'
    },
    'garlic': {
        'name_vi': 'Tỏi',
        'calories': 149,
        'vitamins': ['Vitamin C', 'Vitamin B6'],
        'benefits': 'Kháng khuẩn, tốt cho tim mạch'
    },
    'ginger': {
        'name_vi': 'Gừng',
        'calories': 80,
        'vitamins': ['Vitamin B6', 'Vitamin C'],
        'benefits': 'Chống viêm, hỗ trợ tiêu hóa'
    },
    'grapes': {
        'name_vi': 'Nho',
        'calories': 69,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Chống oxy hóa, tốt cho tim mạch'
    },
    'jalepeno': {
        'name_vi': 'Ớt jalapeño',
        'calories': 29,
        'vitamins': ['Vitamin C', 'Vitamin B6'],
        'benefits': 'Tăng cường trao đổi chất, hỗ trợ tiêu hóa'
    },
    'kiwi': {
        'name_vi': 'Kiwi',
        'calories': 41,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Tăng cường miễn dịch, tốt cho tiêu hóa'
    },
    'lemon': {
        'name_vi': 'Chanh',
        'calories': 29,
        'vitamins': ['Vitamin C', 'Vitamin B6'],
        'benefits': 'Tăng sức đề kháng, giải độc'
    },
    'lettuce': {
        'name_vi': 'Rau diếp',
        'calories': 15,
        'vitamins': ['Vitamin A', 'Vitamin K'],
        'benefits': 'Tốt cho mắt, hỗ trợ tiêu hóa'
    },
    'mango': {
        'name_vi': 'Xoài',
        'calories': 60,
        'vitamins': ['Vitamin A', 'Vitamin C'],
        'benefits': 'Tốt cho mắt, tăng cường miễn dịch'
    },
    'onion': {
        'name_vi': 'Hành tây',
        'calories': 40,
        'vitamins': ['Vitamin C', 'Vitamin B6'],
        'benefits': 'Kháng viêm, tốt cho tim mạch'
    },
    'orange': {
        'name_vi': 'Cam',
        'calories': 47,
        'vitamins': ['Vitamin C', 'Vitamin A'],
        'benefits': 'Tăng cường miễn dịch, tốt cho da'
    },
    'paprika': {
        'name_vi': 'Ớt paprika',
        'calories': 282,
        'vitamins': ['Vitamin A', 'Vitamin E'],
        'benefits': 'Chống oxy hóa, tốt cho mắt'
    },
    'pear': {
        'name_vi': 'Lê',
        'calories': 57,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Tốt cho tiêu hóa, hỗ trợ giảm cân'
    },
    'peas': {
        'name_vi': 'Đậu Hà Lan',
        'calories': 81,
        'vitamins': ['Vitamin K', 'Vitamin C'],
        'benefits': 'Tốt cho xương, hỗ trợ tiêu hóa'
    },
    'pineapple': {
        'name_vi': 'Dứa',
        'calories': 50,
        'vitamins': ['Vitamin C', 'Vitamin B1'],
        'benefits': 'Tăng cường miễn dịch, hỗ trợ tiêu hóa'
    },
    'pomegranate': {
        'name_vi': 'Lựu',
        'calories': 83,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Chống oxy hóa, tốt cho tim mạch'
    },
    'potato': {
        'name_vi': 'Khoai tây',
        'calories': 77,
        'vitamins': ['Vitamin C', 'Vitamin B6'],
        'benefits': 'Cung cấp năng lượng, tốt cho tiêu hóa'
    },
    'raddish': {
        'name_vi': 'Củ cải',
        'calories': 16,
        'vitamins': ['Vitamin C'],
        'benefits': 'Giải độc, hỗ trợ tiêu hóa'
    },
    'soy beans': {
        'name_vi': 'Đậu nành',
        'calories': 173,
        'vitamins': ['Vitamin K', 'Folate'],
        'benefits': 'Giàu protein, tốt cho tim mạch'
    },
    'spinach': {
        'name_vi': 'Rau chân vịt',
        'calories': 23,
        'vitamins': ['Vitamin K', 'Vitamin A'],
        'benefits': 'Tốt cho máu, chống oxy hóa'
    },
    'sweetcorn': {
        'name_vi': 'Ngô ngọt',
        'calories': 86,
        'vitamins': ['Vitamin B', 'Vitamin C'],
        'benefits': 'Cung cấp năng lượng, tốt cho mắt'
    },
    'sweetpotato': {
        'name_vi': 'Khoai lang',
        'calories': 86,
        'vitamins': ['Vitamin A', 'Vitamin C'],
        'benefits': 'Tốt cho tiêu hóa, hỗ trợ giảm cân'
    },
    'tomato': {
        'name_vi': 'Cà chua',
        'calories': 18,
        'vitamins': ['Vitamin C', 'Vitamin K', 'Lycopene'],
        'benefits': 'Chống ung thư, tốt cho tim mạch'
    },
    'turnip': {
        'name_vi': 'Củ cải trắng',
        'calories': 28,
        'vitamins': ['Vitamin C', 'Vitamin K'],
        'benefits': 'Tốt cho tiêu hóa, hỗ trợ giảm cân'
    },
    'watermelon': {
        'name_vi': 'Dưa hấu',
        'calories': 30,
        'vitamins': ['Vitamin A', 'Vitamin C'],
        'benefits': 'Giải khát, tốt cho tim mạch'
    }
}

# Lưu trữ thống kê và lịch sử
detection_stats = {
    'total_detections': 0,
    'unique_fruits': set(),
    'detection_history': [],
    'session_start': datetime.now()
}

# File lưu lịch sử
HISTORY_FILE = 'detection_history.json'

def load_history():
    """Tải lịch sử từ file JSON"""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
            # Chuyển đổi timestamp string về datetime object
            for item in history:
                if isinstance(item.get('timestamp'), str):
                    try:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    except:
                        pass
            return history
    except Exception as e:
        print(f"Lỗi tải lịch sử: {e}")
        return []

def save_history(history):
    """Lưu lịch sử vào file JSON"""
    try:
        # Chuyển đổi datetime objects về string để JSON serializable
        history_to_save = []
        for item in history:
            item_copy = item.copy()
            if isinstance(item_copy.get('timestamp'), datetime):
                item_copy['timestamp'] = item_copy['timestamp'].isoformat()
            history_to_save.append(item_copy)
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi lưu lịch sử: {e}")

# Tải lịch sử khi khởi động
detection_stats['detection_history'] = load_history()

class FruitClassifier:
    def __init__(self, model_path='fruit_classifier_mobilenetv2.h5'):
        """Khởi tạo classifier với mô hình đã huấn luyện"""
        try:
            self.model = load_model(model_path)
            self.img_size = (224, 224)
            print("✅ Đã tải mô hình thành công!")
        except Exception as e:
            print(f"❌ Lỗi tải mô hình: {e}")
            self.model = None
    
    def preprocess_image(self, img):
        """Tiền xử lý ảnh cho mô hình"""
        # Resize ảnh
        img_resized = cv2.resize(img, self.img_size)
        # Chuyển từ BGR sang RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Chuẩn hóa pixel values
        img_array = image.img_to_array(img_rgb)
        img_array = img_array / 255.0
        # Thêm batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, img):
        """Dự đoán loại quả trong ảnh"""
        if self.model is None:
            return "Model not loaded", 0.0
        try:
            # Tiền xử lý ảnh
            processed_img = self.preprocess_image(img)
            # Dự đoán
            predictions = self.model.predict(processed_img)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            # Lấy tên loại quả
            fruit_name = CLASS_NAMES[predicted_class]
            print(f"DEBUG predict: fruit_name={fruit_name}, confidence={confidence}")
            return fruit_name, confidence
        except Exception as e:
            print(f"Lỗi dự đoán: {e}")
            return "Unknown", 0.0

# Khởi tạo classifier
classifier = FruitClassifier()

def update_stats(fruit_name, confidence, img_base64=None, detection_type="upload", user_id=None):
    """Cập nhật thống kê và lưu lịch sử"""
    detection_stats['total_detections'] += 1
    detection_stats['unique_fruits'].add(fruit_name)
    
    # Lấy thông tin dinh dưỡng
    fruit_info = FRUIT_INFO.get(fruit_name, {})
    
    history_item = {
        'id': f"det_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        'fruit_name': fruit_name,
        'fruit_name_vi': fruit_info.get('name_vi', fruit_name),
        'confidence': float(confidence),
        'confidence_percent': f"{float(confidence) * 100:.1f}%",
        'timestamp': datetime.now(),
        'detection_type': detection_type,  # 'upload', 'webcam', 'yolo'
        'user_id': user_id or session.get('username', 'unknown'),
        'fruit_info': fruit_info,
        'image': img_base64,
        'calories': fruit_info.get('calories', 0),
        'vitamins': fruit_info.get('vitamins', []),
        'benefits': fruit_info.get('benefits', '')
    }
    
    detection_stats['detection_history'].append(history_item)
    
    # Giới hạn lịch sử trong memory (giữ 1000 items gần nhất)
    if len(detection_stats['detection_history']) > 1000:
        detection_stats['detection_history'] = detection_stats['detection_history'][-1000:]
    
    # Lưu vào file
    save_history(detection_stats['detection_history'])
    
    return history_item

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'GET':
        return render_template('login.html')
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    users = load_users()
    if username in users and users[username]['password'] == hash_password(password):
        session['logged_in'] = True
        session['username'] = username
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Tên đăng nhập hoặc mật khẩu không đúng!'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Xử lý upload file"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được chọn'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'})
    
    if file:
        try:
            # Đọc ảnh
            img_bytes = file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({'error': 'Không thể đọc file ảnh'})
            
            # Nhận diện
            fruit_name, confidence = classifier.predict(img)
            
            # Lưu ảnh
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Lưu ảnh gốc
            cv2.imwrite(file_path, img)
            
            # Chuyển ảnh sang base64 để hiển thị
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Cập nhật thống kê
            update_stats(fruit_name, confidence, img_base64, detection_type="upload")
            
            # Lấy màu cho loại quả
            color = FRUIT_COLORS.get(fruit_name.lower(), FRUIT_COLORS['default'])
            
            # Lấy thông tin dinh dưỡng
            fruit_info = FRUIT_INFO.get(fruit_name, {})
            
            return jsonify({
                'success': True,
                'fruit_name': fruit_name,
                'fruit_name_vi': fruit_info.get('name_vi', fruit_name),
                'confidence': float(confidence),
                'confidence_percent': f"{confidence:.1%}",
                'image': img_base64,
                'color': color,
                'fruit_info': fruit_info,
                'model_used': 'local',
                'image_path': filename,
                'stats': {
                    'total_detections': detection_stats['total_detections'],
                    'unique_fruits': len(detection_stats['unique_fruits'])
                }
            })
        
        except Exception as e:
            return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'})

@app.route('/webcam', methods=['POST'])
@login_required
def webcam_detection():
    """Xử lý ảnh từ webcam"""
    try:
        # Lấy dữ liệu base64 từ webcam
        data = request.get_json()
        img_data = data['image'].split(',')[1]  # Bỏ qua phần "data:image/jpeg;base64,"
        
        # Chuyển base64 sang numpy array
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Không thể đọc ảnh từ webcam'})
        
        # Nhận diện
        fruit_name, confidence = classifier.predict(img)
        
        # Chuyển ảnh sang base64 để hiển thị
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Cập nhật thống kê
        update_stats(fruit_name, confidence, img_base64, detection_type="webcam")
        
        # Lấy màu cho loại quả
        color = FRUIT_COLORS.get(fruit_name.lower(), FRUIT_COLORS['default'])
        
        # Lấy thông tin dinh dưỡng
        fruit_info = FRUIT_INFO.get(fruit_name, {})
        
        return jsonify({
            'success': True,
            'fruit_name': fruit_name,
            'fruit_name_vi': fruit_info.get('name_vi', fruit_name),
            'confidence': float(confidence),
            'confidence_percent': f"{confidence:.1%}",
            'image': img_base64,
            'color': color,
            'fruit_info': fruit_info,
            'model_used': 'local',
            'stats': {
                'total_detections': detection_stats['total_detections'],
                'unique_fruits': len(detection_stats['unique_fruits'])
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý webcam: {str(e)}'})

@app.route('/stats')
@login_required
def get_stats():
    session_duration = (datetime.now() - detection_stats['session_start']).total_seconds()
    minutes = int(session_duration // 60)
    seconds = int(session_duration % 60)
    # Tính trung bình confidence
    history = detection_stats['detection_history']
    if history:
        avg_conf = sum(h['confidence'] for h in history) / len(history)
        accuracy_rate = f"{avg_conf * 100:.0f}%"
    else:
        accuracy_rate = "0%"
    return jsonify({
        'total_detections': detection_stats['total_detections'],
        'unique_fruits': len(detection_stats['unique_fruits']),
        'session_time': f"{minutes:02d}:{seconds:02d}",
        'accuracy_rate': accuracy_rate
    })

@app.route('/history')
@login_required
def get_history():
    """Lấy lịch sử nhận diện với phân trang và lọc"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    search = request.args.get('search', '').lower()
    fruit_filter = request.args.get('fruit', '').lower()
    date_filter = request.args.get('date', '')
    
    # Lọc lịch sử
    filtered_history = detection_stats['detection_history']
    
    # Lọc theo tìm kiếm
    if search:
        filtered_history = [
            h for h in filtered_history 
            if search in h.get('fruit_name', '').lower() or 
               search in h.get('fruit_name_vi', '').lower()
        ]
    
    # Lọc theo loại quả
    if fruit_filter:
        filtered_history = [
            h for h in filtered_history 
            if fruit_filter in h.get('fruit_name', '').lower()
        ]
    
    # Lọc theo ngày
    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, '%Y-%m-%d').date()
            filtered_history = [
                h for h in filtered_history 
                if isinstance(h.get('timestamp'), datetime) and 
                   h['timestamp'].date() == filter_date
            ]
        except:
            pass
    
    # Sắp xếp theo thời gian mới nhất
    filtered_history.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
    
    # Phân trang
    total_items = len(filtered_history)
    total_pages = (total_items + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_history = filtered_history[start_idx:end_idx]
    
    # Chuyển đổi datetime về string cho JSON
    history_for_json = []
    for item in paginated_history:
        item_copy = item.copy()
        if isinstance(item_copy.get('timestamp'), datetime):
            item_copy['timestamp'] = item_copy['timestamp'].isoformat()
        history_for_json.append(item_copy)
    
    return jsonify({
        'history': history_for_json,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        },
        'filters': {
            'search': search,
            'fruit': fruit_filter,
            'date': date_filter
        }
    })

@app.route('/fruit-info/<fruit_name>')
@login_required
def get_fruit_info(fruit_name):
    """Lấy thông tin chi tiết về loại quả"""
    fruit_info = FRUIT_INFO.get(fruit_name, {})
    if fruit_info:
        return jsonify({'success': True, 'info': fruit_info})
    else:
        return jsonify({'success': False, 'error': 'Không tìm thấy thông tin về loại quả này'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Phục vụ file đã upload"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete_history', methods=['POST'])
@login_required
def delete_history():
    data = request.get_json()
    item_id = data.get('id')
    if not item_id:
        return jsonify({'success': False, 'error': 'Thiếu ID của mục lịch sử'})
    
    before = len(detection_stats['detection_history'])
    detection_stats['detection_history'] = [
        h for h in detection_stats['detection_history'] if h.get('id') != item_id
    ]
    after = len(detection_stats['detection_history'])
    
    if after < before:
        # Lưu vào file sau khi xóa
        save_history(detection_stats['detection_history'])
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Không tìm thấy mục lịch sử để xóa'})

@app.route('/delete_all_history', methods=['POST'])
@login_required
def delete_all_history():
    detection_stats['detection_history'] = []
    detection_stats['unique_fruits'] = set()
    detection_stats['total_detections'] = 0
    detection_stats['session_start'] = datetime.now()
    
    # Lưu vào file sau khi xóa tất cả
    save_history(detection_stats['detection_history'])
    
    return jsonify({'success': True})

@app.route('/export_history_excel')
@login_required
def export_history_excel():
    """Xuất lịch sử nhận diện ra file Excel"""
    try:
        # Tạo DataFrame từ lịch sử
        history_data = []
        for item in detection_stats['detection_history']:
            history_data.append({
                'Tên quả': item['fruit_name'],
                'Độ tin cậy': f"{item['confidence']:.2f}",
                'Thời gian': item['timestamp']
            })
        
        df = pd.DataFrame(history_data)
        
        # Tạo file Excel
        excel_file = 'fruit_detection_history.xlsx'
        df.to_excel(excel_file, index=False, engine='openpyxl')
        
        return send_file(excel_file, as_attachment=True, download_name='fruit_detection_history.xlsx')
    
    except Exception as e:
        return jsonify({'error': f'Lỗi xuất file: {str(e)}'})

def open_browser():
    """Hàm mở trình duyệt"""
    time.sleep(1.5)  # Đợi server khởi động
    webbrowser.open('http://localhost:5000')

USERS_FILE = 'users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({'success': False, 'error': 'Vui lòng nhập đầy đủ thông tin!'})
    users = load_users()
    if username in users:
        return jsonify({'success': False, 'error': 'Tên đăng nhập đã tồn tại!'})
    users[username] = {'password': hash_password(password)}
    save_users(users)
    return jsonify({'success': True})

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')
    data = request.get_json()
    username = data.get('username', '').strip()
    users = load_users()
    if username not in users:
        return jsonify({'success': False, 'error': 'Tên đăng nhập không tồn tại!'})
    # Đặt lại mật khẩu mặc định (ví dụ: 123456) và thông báo cho người dùng
    new_password = '123456'
    users[username]['password'] = hash_password(new_password)
    save_users(users)
    return jsonify({'success': True, 'message': f'Mật khẩu mới của bạn là: {new_password}. Vui lòng đổi lại sau khi đăng nhập.'})

class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Khởi tạo YOLOv8 detector
        model_path: đường dẫn đến model YOLOv8 (có thể là yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLOv8 model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            # Fallback to nano model if custom model fails
            try:
                self.model = YOLO('yolov8n.pt')
                print("Using default YOLOv8n model")
            except Exception as e2:
                print(f"Failed to load any YOLOv8 model: {e2}")
                self.model = None
    
    def detect_fruits(self, img, conf_threshold=0.25, iou_threshold=0.45):
        """
        Nhận diện và đếm quả trong ảnh
        Args:
            img: ảnh đầu vào (numpy array hoặc PIL Image)
            conf_threshold: ngưỡng confidence
            iou_threshold: ngưỡng IoU cho NMS
        Returns:
            dict: kết quả nhận diện với thông tin đếm và bounding boxes
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'YOLOv8 model not loaded',
                'counts': {},
                'total_count': 0,
                'annotated_image': None
            }
        
        try:
            # Chuyển đổi ảnh nếu cần
            if isinstance(img, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                return {
                    'success': False,
                    'error': 'Unsupported image format',
                    'counts': {},
                    'total_count': 0,
                    'annotated_image': None
                }
            
            # Thực hiện detection
            results = self.model(pil_img, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            # Xử lý kết quả
            detections = []
            counts = defaultdict(int)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Lấy thông tin detection
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Chỉ đếm các loại quả/rau củ
                        if class_name.lower() in [name.lower() for name in CLASS_NAMES]:
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'class_name': class_name,
                                'class_id': class_id
                            })
                            counts[class_name] += 1
            
            # Tạo ảnh có annotation
            annotated_img = self._draw_detections(pil_img, detections)
            
            return {
                'success': True,
                'counts': dict(counts),
                'total_count': sum(counts.values()),
                'detections': detections,
                'annotated_image': annotated_img
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'counts': {},
                'total_count': 0,
                'annotated_image': None
            }
    
    def _draw_detections(self, img, detections):
        """
        Vẽ bounding boxes và labels lên ảnh
        """
        img_array = np.array(img)
        img_draw = img_array.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Chuyển về int để vẽ
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Lấy màu cho class
            color = FRUIT_COLORS.get(class_name.lower(), FRUIT_COLORS['default'])
            # Chuyển hex color sang BGR
            color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))[::-1]
            
            # Vẽ bounding box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Vẽ label
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color_bgr, -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return Image.fromarray(img_draw)

# Khởi tạo YOLOv8 detector
yolo_detector = YOLOv8Detector()

@app.route('/yolo-detect', methods=['POST'])
@login_required
def yolo_detection():
    """
    Nhận diện và đếm quả bằng YOLOv8
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Không có file được upload'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Không có file được chọn'})
        
        if file:
            # Đọc ảnh
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Thực hiện detection bằng YOLOv8
            result = yolo_detector.detect_fruits(img)
            
            if result['success']:
                # Chuyển ảnh có annotation thành base64
                annotated_img = result['annotated_image']
                buffered = io.BytesIO()
                annotated_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Cập nhật thống kê
                for fruit_name, count in result['counts'].items():
                    for _ in range(count):
                        update_stats(fruit_name, 0.9, img_str, detection_type="yolo")
                
                return jsonify({
                    'success': True,
                    'counts': result['counts'],
                    'total_count': result['total_count'],
                    'detections': result['detections'],
                    'annotated_image': img_str,
                    'message': f'Đã phát hiện {result["total_count"]} quả/rau củ'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Lỗi không xác định')
                })
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi: {str(e)}'})

@app.route('/history-stats')
@login_required
def get_history_stats():
    """Lấy thống kê chi tiết về lịch sử nhận diện"""
    history = detection_stats['detection_history']
    
    # Thống kê theo loại quả
    fruit_counts = {}
    fruit_confidence_avg = {}
    
    # Thống kê theo ngày
    daily_counts = {}
    
    # Thống kê theo loại detection
    detection_type_counts = {}
    
    for item in history:
        fruit_name = item.get('fruit_name', 'Unknown')
        confidence = item.get('confidence', 0)
        timestamp = item.get('timestamp')
        detection_type = item.get('detection_type', 'unknown')
        
        # Đếm theo loại quả
        if fruit_name not in fruit_counts:
            fruit_counts[fruit_name] = 0
            fruit_confidence_avg[fruit_name] = []
        fruit_counts[fruit_name] += 1
        fruit_confidence_avg[fruit_name].append(confidence)
        
        # Đếm theo ngày
        if isinstance(timestamp, datetime):
            date_str = timestamp.strftime('%Y-%m-%d')
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
        
        # Đếm theo loại detection
        detection_type_counts[detection_type] = detection_type_counts.get(detection_type, 0) + 1
    
    # Tính trung bình confidence cho mỗi loại quả
    for fruit_name in fruit_confidence_avg:
        if fruit_confidence_avg[fruit_name]:
            fruit_confidence_avg[fruit_name] = sum(fruit_confidence_avg[fruit_name]) / len(fruit_confidence_avg[fruit_name])
    
    # Sắp xếp theo số lượng giảm dần
    top_fruits = sorted(fruit_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Sắp xếp theo ngày
    daily_stats = sorted(daily_counts.items(), key=lambda x: x[0], reverse=True)[:30]
    
    return jsonify({
        'total_detections': len(history),
        'unique_fruits': len(fruit_counts),
        'top_fruits': [
            {
                'name': fruit_name,
                'name_vi': FRUIT_INFO.get(fruit_name, {}).get('name_vi', fruit_name),
                'count': count,
                'avg_confidence': fruit_confidence_avg.get(fruit_name, 0)
            }
            for fruit_name, count in top_fruits
        ],
        'daily_stats': [
            {
                'date': date,
                'count': count
            }
            for date, count in daily_stats
        ],
        'detection_types': detection_type_counts,
        'recent_activity': {
            'today': daily_counts.get(datetime.now().strftime('%Y-%m-%d'), 0),
            'yesterday': daily_counts.get((datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), 0),
            'this_week': sum(
                daily_counts.get((datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 0)
                for i in range(7)
            )
        }
    })

if __name__ == '__main__':
    print("🚀 Khởi động hệ thống nhận diện quả thông minh...")
    print("📱 Tự động mở trình duyệt trong vài giây...")
    print("🎯 Tính năng:")
    print("   - Dark mode / Light mode")
    print("   - Thống kê real-time")
    print("   - Thông tin dinh dưỡng")
    print("   - Lịch sử nhận diện")
    print("   - Giao diện responsive")
    
    # Tạo thread để mở trình duyệt
    threading.Thread(target=open_browser).start()
    
    # Khởi động Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)