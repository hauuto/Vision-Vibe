from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/essay1')
def essay_1():
    return render_template('essay1.html')

# --- API DEMOS ---

@app.route('/api/hello')
def api_hello():
    return jsonify({'message': 'Hello from Flask API!'})

@app.route('/api/opencv-demo', methods=['POST'])
def api_opencv_demo():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    in_memory = BytesIO()
    file.save(in_memory)
    data = np.frombuffer(in_memory.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    # Simple OpenCV processing: convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, buf = cv2.imencode('.png', gray)
    img_b64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({'image': img_b64})

# Template context processor để có thể sử dụng include trong template
@app.context_processor
def inject_template_vars():
    return dict()

if __name__ == '__main__':
    app.run(debug=True)
