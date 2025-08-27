from flask_frozen import Freezer
from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import sys

# Add the tieuluan directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tieuluan'))

# Import methods from main.py
try:
    from main import (
        apply_log_transform,
        apply_power_transform, 
        apply_negative_transform,
        histogram_equalization,
        apply_piecewise_linear,
        apply_CLAHE
    )
except ImportError as e:
    print(f"Error importing from main.py: {e}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slide1/')
def slide01():
    return render_template('components/slide01.html')

@app.route('/test')
def test():
    return render_template('test.html')
# --- API DEMOS ---

@app.route('/api/hello/')
def api_hello():
    return jsonify({'message': 'Hello from Flask API!'})

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        # Get the uploaded file or base64 image
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Convert file to OpenCV image
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif request.json and 'image_data' in request.json:
            # Handle base64 image data
            image_data = request.json['image_data']
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 to image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get processing options from request
        method = request.form.get('method') if request.form else request.json.get('method', 'negative')
        grayscale = request.form.get('grayscale') if request.form else request.json.get('grayscale', 'false')
        
        # Convert grayscale string to boolean
        is_grayscale = grayscale.lower() in ['true', '1', 'on', 'yes']
        
        # Convert to grayscale if requested
        if is_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Convert back to 3-channel for consistency in processing
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Process image using methods from main.py
        if method == '1':
            processed_img = apply_log_transform(img)
        elif method == '2':
            gamma = float(request.form.get('gamma', 0.5)) if request.form else float(request.json.get('gamma', 0.5))
            processed_img = apply_power_transform(img, gamma)
        elif method == '3':
            processed_img = apply_negative_transform(img)
        elif method == '4':
            processed_img = histogram_equalization(img)
        elif method == '5':
            # Get piecewise linear parameters
            r1 = int(request.form.get('r1', 100)) if request.form else int(request.json.get('r1', 100))
            s1 = int(request.form.get('s1', 50)) if request.form else int(request.json.get('s1', 50))
            r2 = int(request.form.get('r2', 200)) if request.form else int(request.json.get('r2', 200))
            s2 = int(request.form.get('s2', 200)) if request.form else int(request.json.get('s2', 200))
            processed_img = apply_piecewise_linear(img, r1, s1, r2, s2)
        elif method =='6':
            processed_img = apply_CLAHE(img)
        else:
            # Default to negative transform
            processed_img = apply_negative_transform(img)
        
        # Convert processed image back to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{img_base64}',
            'method_used': method,
            'grayscale_applied': is_grayscale,
            'original_shape': img.shape,
            'processed_shape': processed_img.shape
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/available-methods')
def available_methods():
    """Return list of available image processing methods"""
    methods = [
        {'name': 'negative', 'description': 'Negative Transformation (Âm bản)'},
        {'name': 'log_transform', 'description': 'Log Transformation (Chuyển đổi log)'},
        {'name': 'power_transform', 'description': 'Power/Gamma Transformation (Chuyển đổi mũ)'},
        {'name': 'histogram_equalization', 'description': 'Histogram Equalization (Cân bằng histogram)'}
    ]
    return jsonify({'methods': methods})

# Template context processor để có thể sử dụng include trong template
@app.context_processor
def inject_template_vars():
    return dict()

freezer = Freezer(app)
app.config['FREEZER_DESTINATION'] = 'docs'
app.config['FREEZER_BASE_URL'] = ''
app.config['FREEZER_RELATIVE_URLS'] = True
if __name__ == '__main__':
    freezer.freeze()
    app.run(debug=True)
