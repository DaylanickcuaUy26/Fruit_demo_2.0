import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

class FruitApp:
    def __init__(self):
        self.model = load_model('fruit_classifier_mobilenetv2.h5')
        self.class_names = [
            'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
            'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
            'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
            'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
            'sweetpotato', 'tomato', 'turnip', 'watermelon'
        ]
        self.yolo = YOLO('yolov8n.pt')
        self.root = tk.Tk()
        self.root.title('Fruit Recognition')
        self.root.geometry('750x600')
        tk.Button(self.root, text='Nhận diện ảnh', command=self.detect_image, bg='green', fg='white').pack(pady=10)
        tk.Button(self.root, text='Nhận diện webcam', command=self.detect_webcam, bg='blue', fg='white').pack(pady=5)
        self.image_label = tk.Label(self.root, bg='white', width=60, height=20)
        self.image_label.pack(pady=10)
        self.result_label = tk.Label(self.root, text='')
        self.result_label.pack(pady=10)

    def detect_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.png')])
        if not file_path:
            return
        img = cv2.imread(file_path)
        img_disp = self.show_img(img)
        self.image_label.config(image=img_disp)
        self.image_label.image = img_disp
        # Mobilenetv2
        img_resized = cv2.resize(img, (224,224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        arr = image.img_to_array(img_rgb)/255.0
        arr = np.expand_dims(arr, 0)
        pred = self.model.predict(arr)
        idx = np.argmax(pred[0])
        conf = pred[0][idx]
        fruit = self.class_names[idx]
        self.result_label.config(text=f'Kết quả: {fruit}\nĐộ tin cậy: {conf:.2%}')

    def detect_webcam(self):
        threading.Thread(target=self._detect_webcam, daemon=True).start()

    def _detect_webcam(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.yolo(frame)
            annotated = results[0].plot()
            cv2.imshow('YOLO Webcam - Nhấn Q để thoát', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def show_img(self, img):
        h, w = img.shape[:2]
        max_size = 450
        if w > max_size or h > max_size:
            scale = min(max_size/w, max_size/h)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        bg = Image.new('RGB', (500, 350), (255,255,255))
        img_w, img_h = pil_img.size
        offset = ((500 - img_w)//2, (350 - img_h)//2)
        bg.paste(pil_img, offset)
        return ImageTk.PhotoImage(bg)

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    FruitApp().run() 