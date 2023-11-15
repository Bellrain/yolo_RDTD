import pandas as pd
from PIL import Image
import io
import base64
from flask import Flask, request, render_template
import torch
from torchvision.transforms import functional as F
from PIL import Image
import ultralytics
from ultralytics.models.yolo.model import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
app = Flask(__name__)

# YOLO 모델 초기화 및 가중치 로드
# YOLO 모델 초기화
Yolo_model = YOLO("best_final.pt")



def process_and_get_result_image(input_image):
    result = Yolo_model.predict(input_image)
    PBOX = pd.DataFrame(columns=range(6))
    for i in range(len(result)):
        arri = pd.DataFrame(result[i].boxes.data.cpu().numpy()).astype(float)
        PBOX = pd.concat([PBOX, arri], axis=0)
    PBOX.columns = ['x', 'y', 'x2', 'y2', 'confidence', 'class']
    image = cv2.imread(input_image)
    for i in range(len(PBOX)):
        x = int(PBOX['x'][i])
        y = int(PBOX['y'][i])
        x2 = int(PBOX['x2'][i])
        y2 = int(PBOX['y2'][i])
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        label = f"Damage: {PBOX['confidence'][i]:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No image part'

    file = request.files['image']

    if file:
        with torch.no_grad():
            # 이미지 처리 및 예측
            #result_image = process_and_get_result_image(file) # 이 부분은 이미지 처리 및 결과를 가져오는 로직으로 대체
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"image_{current_time}.jpg"
            image = Image.open(file)
            image.save(filename,'JPEG')
            result_image = process_and_get_result_image(filename)
            # Matplotlib로 이미지 그리기
            plt.imshow(result_image)
            plt.axis('off')  # 이미지 테두리 제거


            # 그린 이미지를 바이트로 저장
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            plt.clf()
            img_buf.seek(0)
            img_data = base64.b64encode(img_buf.read()).decode()
            

            return render_template('index.html', result=img_data)

            # OpenCV 이미지를 PIL 이미지로 변환 (BGR -> RGB)
            #result_image = cv2.cvtColor("image.jpg", cv2.COLOR_BGR2RGB)
            #result_image_pil = Image.fromarray(result_image)

            # PIL 이미지를 Base64로 인코딩
            #buffered = io.BytesIO()
            #result_image_pil.save(buffered, format="JPEG")
            #result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            # 이미지를 Base64로 인코딩




if __name__ == '__main__':
    app.run(debug=True)