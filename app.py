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
    from myself_apply import (
        apply_log_transform,
        apply_power_transform, 
        apply_negative_transform,
        histogram_equalization,
        apply_piecewise_linear,
        apply_CLAHE
    )
    from cv2_apply import (
        cv2_apply_log_transform,
        cv2_apply_power_transform,
        cv2_apply_negative_transform,
        cv2_histogram_equalization,
        cv2_apply_piecewise_linear,
        cv2_apply_CLAHE,
        compare_images
    )
except ImportError as e:
    print(f"Error importing from modules: {e}")

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
        method = request.form.get('method') if request.form else request.json.get('method', '3')
        grayscale = request.form.get('grayscale') if request.form else request.json.get('grayscale', 'false')
        show_cv2_comparison = request.form.get('use_cv2') if request.form else request.json.get('use_cv2', 'false')
        
        # Convert string to boolean
        is_grayscale = grayscale.lower() in ['true', '1', 'on', 'yes']
        is_show_cv2_comparison = show_cv2_comparison.lower() in ['true', '1', 'on', 'yes']
        
        # Convert to grayscale if requested
        if is_grayscale:
            print(f"🔍 Converting to grayscale. Original shape: {img.shape}")
            print(f"Original range: [{np.min(img)}, {np.max(img)}]")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"After grayscale: {img.shape}, range: [{np.min(img)}, {np.max(img)}]")
            
            # Convert back to 3-channel for consistency in processing
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            print(f"After BGR conversion: {img.shape}, range: [{np.min(img)}, {np.max(img)}]")
        
        # Get additional parameters
        gamma = float(request.form.get('gamma', 0.5)) if request.form else float(request.json.get('gamma', 0.5))
        r1 = int(request.form.get('r1', 100)) if request.form else int(request.json.get('r1', 100))
        s1 = int(request.form.get('s1', 50)) if request.form else int(request.json.get('s1', 50))
        r2 = int(request.form.get('r2', 200)) if request.form else int(request.json.get('r2', 200))
        s2 = int(request.form.get('s2', 200)) if request.form else int(request.json.get('s2', 200))
        
        # LUÔN sử dụng custom method từ myself_apply.py làm kết quả chính
        processed_img = None
        cv2_processed_img = None
        comparison_metrics = None
        
        if method == '1':  # Log transform
            processed_img = apply_log_transform(img)
            if is_show_cv2_comparison:
                cv2_processed_img = cv2_apply_log_transform(img)
                comparison_metrics = compare_images(processed_img, cv2_processed_img)
                
        elif method == '2':  # Power transform
            processed_img = apply_power_transform(img, gamma)
            if is_show_cv2_comparison:
                cv2_processed_img = cv2_apply_power_transform(img, gamma)
                comparison_metrics = compare_images(processed_img, cv2_processed_img)
                
        elif method == '3':  # Negative
            processed_img = apply_negative_transform(img)
            if is_show_cv2_comparison:
                cv2_processed_img = cv2_apply_negative_transform(img)
                comparison_metrics = compare_images(processed_img, cv2_processed_img)
                
        elif method == '4':  # Histogram equalization
            processed_img = histogram_equalization(img)
            if is_show_cv2_comparison:
                cv2_processed_img = cv2_histogram_equalization(img)
                comparison_metrics = compare_images(processed_img, cv2_processed_img)
                
        elif method == '5':  # Piecewise linear
            processed_img = apply_piecewise_linear(img, r1, s1, r2, s2)
            if is_show_cv2_comparison:
                cv2_processed_img = cv2_apply_piecewise_linear(img, r1, s1, r2, s2)
                comparison_metrics = compare_images(processed_img, cv2_processed_img)
                
        elif method == '6':  # CLAHE
            processed_img = apply_CLAHE(img)
            if is_show_cv2_comparison:
                cv2_processed_img = cv2_apply_CLAHE(img)
                comparison_metrics = compare_images(processed_img, cv2_processed_img)
        else:
            # Default to negative transform
            processed_img = apply_negative_transform(img)
            if is_show_cv2_comparison:
                cv2_processed_img = cv2_apply_negative_transform(img)
                comparison_metrics = compare_images(processed_img, cv2_processed_img)
        
        # Convert processed image back to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response_data = {
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{img_base64}',
            'method_used': method,
            'grayscale_applied': is_grayscale,
            'show_cv2_comparison': is_show_cv2_comparison,
            'original_shape': list(img.shape),
            'processed_shape': list(processed_img.shape)
        }
        
        # Add CV2 comparison data if checkbox was ticked
        if is_show_cv2_comparison and cv2_processed_img is not None:
            _, cv2_buffer = cv2.imencode('.jpg', cv2_processed_img)
            cv2_img_base64 = base64.b64encode(cv2_buffer).decode('utf-8')
            response_data['cv2_processed_image'] = f'data:image/jpeg;base64,{cv2_img_base64}'
            response_data['comparison_metrics'] = comparison_metrics
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(f"Error in process_image: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/available-methods')
def available_methods():
    """Return list of available image processing methods"""
    methods = [
        {'name': '3', 'description': 'Negative Transformation (Âm bản)'},
        {'name': '1', 'description': 'Log Transformation (Chuyển đổi log)'},
        {'name': '2', 'description': 'Power/Gamma Transformation (Chuyển đổi mũ)'},
        {'name': '4', 'description': 'Histogram Equalization (Cân bằng histogram)'},
        {'name': '5', 'description': 'Piecewise Linear Transform (Chuyển đổi đoạn)'},
        {'name': '6', 'description': 'CLAHE (Contrast Limited Adaptive Histogram Equalization)'}
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
    app.run(debug=True)