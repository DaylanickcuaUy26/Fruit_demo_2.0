#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import importlib.util

def check_model():
    """Kiểm tra mô hình đã huấn luyện"""
    if not os.path.exists('fruit_classifier_mobilenetv2.h5'):
        print("❌ Không tìm thấy mô hình đã huấn luyện!")
        print("Vui lòng chạy 'python train_model.py' trước.")
        return False
    return True

def check_libraries():
    """Kiểm tra thư viện cần thiết"""
    required_libs = ['flask', 'tensorflow', 'cv2', 'numpy', 'PIL']
    missing_libs = []
    
    for lib in required_libs:
        if lib == 'cv2':
            spec = importlib.util.find_spec('cv2')
        elif lib == 'PIL':
            spec = importlib.util.find_spec('PIL')
        else:
            spec = importlib.util.find_spec(lib)
        
        if spec is None:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"❌ Thiếu thư viện: {', '.join(missing_libs)}")
        print("Đang cài đặt...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Cài đặt thành công!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Lỗi cài đặt thư viện!")
            return False
    
    return True

def main():
    """Hàm chính"""
    print("=" * 50)
    print("    HỆ THỐNG NHẬN DIỆN QUẢ AI")
    print("=" * 50)
    print()
    
    # Kiểm tra mô hình
    print("🔍 Đang kiểm tra mô hình...")
    if not check_model():
        input("Nhấn Enter để thoát...")
        return
    
    print("✅ Mô hình đã sẵn sàng!")
    print()
    
    # Kiểm tra thư viện
    print("🔍 Đang kiểm tra thư viện...")
    if not check_libraries():
        input("Nhấn Enter để thoát...")
        return
    
    print("✅ Thư viện đã sẵn sàng!")
    print()
    
    # Khởi động web app
    print("🚀 Khởi động web application...")
    print("📱 Trình duyệt sẽ tự động mở trong vài giây...")
    print()
    print("Nhấn Ctrl+C để dừng chương trình")
    print()
    
    try:
        # Import và chạy app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        input("Nhấn Enter để thoát...")

if __name__ == "__main__":
    main() 