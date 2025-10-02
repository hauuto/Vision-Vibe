from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import methods from vision_core package
try:
    from vision_core.myself_apply import (
        apply_log_transform,
        apply_power_transform, 
        apply_negative_transform,
        histogram_equalization,
        apply_piecewise_linear,
        apply_CLAHE
    )
    from vision_core.cv2_apply import (
        cv2_apply_log_transform,
        cv2_apply_power_transform,
        cv2_apply_negative_transform,
        cv2_histogram_equalization,
        cv2_apply_piecewise_linear,
        cv2_apply_CLAHE,
        compare_images
    )
    from vision_core.filters import process_operation, process_operation_cv2
except ImportError as e:
    print(f"Error importing from modules: {e}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slide01/')
def slide01():
    return render_template('slide01.html')

@app.route('/point_processing/')
def test():
    return render_template('point_processing.html')

# New page for filtering and edges
@app.route('/filter/')
def filter_page():
    return render_template('filter.html')

# --- API DEMOS ---
@app.route('/api/hello/')
def api_hello():
    return jsonify({'message': 'Hello from Flask API!'})

# Utility to encode image to base64 PNG
_def_png = '.png'

def _encode_image(img: np.ndarray) -> str:
    if img is None:
        raise ValueError('encode_image: None image')
    # If single channel, ensure proper shape
    enc_img = img
    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        pass
    else:
        raise ValueError(f'Unsupported image shape for encoding: {img.shape}')
    success, buffer = cv2.imencode(_def_png, enc_img)
    if not success:
        raise RuntimeError('Failed to encode image')
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode('utf-8')

@app.route('/api/vision/process', methods=['POST'])
def api_vision_process():
    try:
        # Accept image via multipart form or JSON base64
        img = None
        params = {}
        op = None
        engine = 'custom'

        if 'image' in request.files:
            file = request.files['image']
            if not file or file.filename == '':
                return jsonify({'error': 'No file provided'}), 400
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            op = request.form.get('op', '')
            engine = (request.form.get('engine') or 'custom').lower()
            # Collect all params from form
            params = {k: v for k, v in request.form.items() if k not in ('op', 'engine')}
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get('image_data')
            op = data.get('op', '')
            engine = (data.get('engine') or 'custom').lower()
            params = data.get('params', {}) or {}
            if b64:
                if b64.startswith('data:image'):
                    b64 = b64.split(',')[1]
                image_bytes = base64.b64decode(b64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            return jsonify({'error': 'No image provided or decode failed'}), 400
        if img.ndim == 2:
            pass
        elif img.ndim == 3 and img.shape[2] in (3, 4):
            pass
        else:
            # Convert to BGR if single channel expanded
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Execute operation (choose engine)
        if engine == 'cv2':
            out = process_operation_cv2(img, op, params)
        else:
            out = process_operation(img, op, params)

        resp = {
            'success': True,
            'op': op,
            'engine': engine,
        }
        # Encode main output
        if 'output' in out:
            resp['output'] = _encode_image(out['output'])
        # Encode intermediates when present
        if 'grad_x' in out:
            resp['grad_x'] = _encode_image(out['grad_x'])
        if 'grad_y' in out:
            resp['grad_y'] = _encode_image(out['grad_y'])
        if 'magnitude' in out:
            resp['magnitude'] = _encode_image(out['magnitude'])
        if 'threshold' in out:
            resp['threshold'] = _encode_image(out['threshold'])
        if 'laplacian' in out:
            resp['laplacian'] = _encode_image(out['laplacian'])

        return jsonify(resp)

    except Exception as e:
        import traceback
        print('api_vision_process error:', e)
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        # Get the uploaded file or base64 image
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Convert file to OpenCV image with better error handling
            try:
                image_data = file.read()
                if len(image_data) == 0:
                    return jsonify({'error': 'Empty file'}), 400
                
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    return jsonify({'error': 'Could not decode image. Please check file format.'}), 400
                    
            except Exception as e:
                return jsonify({'error': f'Error reading image file: {str(e)}'}), 400
            
        elif request.json and 'image_data' in request.json:
            # Handle base64 image data
            try:
                image_data = request.json['image_data']
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decode base64 to image
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    return jsonify({'error': 'Could not decode base64 image data'}), 400
                    
            except Exception as e:
                return jsonify({'error': f'Error processing base64 image: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Validate image dimensions
        if img.shape[0] < 1 or img.shape[1] < 1:
            return jsonify({'error': 'Invalid image dimensions'}), 400
        
        # Get processing options from request with proper validation
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
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(f"After grayscale: {img.shape}, range: [{np.min(img)}, {np.max(img)}]")
                
        
        # Get additional parameters with validation
        try:
            gamma = float(request.form.get('gamma', 0.5)) if request.form else float(request.json.get('gamma', 0.5))
            gamma = max(0.1, min(gamma, 5.0))  # Clamp gamma to reasonable range
            
            r1 = int(request.form.get('r1', 100)) if request.form else int(request.json.get('r1', 100))
            s1 = int(request.form.get('s1', 50)) if request.form else int(request.json.get('s1', 50))
            r2 = int(request.form.get('r2', 200)) if request.form else int(request.json.get('r2', 200))
            s2 = int(request.form.get('s2', 200)) if request.form else int(request.json.get('s2', 200))
            
            # Validate piecewise parameters
            r1 = max(0, min(r1, 255))
            r2 = max(r1, min(r2, 255))
            s1 = max(0, min(s1, 255))
            s2 = max(0, min(s2, 255))
            
        except ValueError as e:
            return jsonify({'error': f'Invalid parameter values: {str(e)}'}), 400
        
        # Process image with better error handling
        processed_img = None
        cv2_processed_img = None
        comparison_metrics = None
        
        try:
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
                print(f"🔧 Applying histogram equalization to image shape: {img.shape}")
                processed_img = histogram_equalization(img)
                print(f"✅ Histogram equalization completed. Result shape: {processed_img.shape}")
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
                    
        except Exception as e:
            import traceback
            print(f"Error during image processing: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Image processing failed: {str(e)}'}), 500
        
        # Validate processed image
        if processed_img is None:
            return jsonify({'error': 'Image processing returned None'}), 500
            
        if processed_img.shape != img.shape:
            print(f"⚠️  Shape mismatch: original {img.shape}, processed {processed_img.shape}")
        
        # Convert processed image back to base64 with error handling
        try:
            success, buffer = cv2.imencode('.jpg', processed_img)
            if not success:
                return jsonify({'error': 'Failed to encode processed image'}), 500
                
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            return jsonify({'error': f'Error encoding image: {str(e)}'}), 500
        
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
            try:
                success, cv2_buffer = cv2.imencode('.jpg', cv2_processed_img)
                if success:
                    cv2_img_base64 = base64.b64encode(cv2_buffer).decode('utf-8')
                    response_data['cv2_processed_image'] = f'data:image/jpeg;base64,{cv2_img_base64}'
                    response_data['comparison_metrics'] = comparison_metrics
                else:
                    print("Warning: Failed to encode CV2 processed image")
            except Exception as e:
                print(f"Warning: Error encoding CV2 image: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(f"Error in process_image: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/available-methods.json')
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

@app.route('/api/vision/pipeline', methods=['POST'])
def api_vision_pipeline():
    try:
        data = request.get_json(silent=True) or {}
        b64 = data.get('image_data')
        engine = (data.get('engine') or 'cv2').lower()
        steps = data.get('steps') or []

        if not b64:
            return jsonify({'error': 'No image_data provided'}), 400
        if not isinstance(steps, list) or len(steps) == 0:
            return jsonify({'error': 'No steps provided'}), 400
        if len(steps) > 50:
            return jsonify({'error': 'Too many steps (max 50)'}), 400

        if b64.startswith('data:image'):
            b64 = b64.split(',')[1]
        image_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        current = img
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                return jsonify({'error': f'Invalid step at index {idx}'}), 400
            op = (step.get('op') or '').lower()
            params = step.get('params') or {}
            if engine == 'cv2':
                out = process_operation_cv2(current, op, params)
            else:
                out = process_operation(current, op, params)
            if 'output' not in out or out['output'] is None:
                return jsonify({'error': f'Operation {op} failed at step {idx}'}), 500
            current = out['output']

        resp = {
            'success': True,
            'engine': engine,
            'steps_count': len(steps),
            'output': _encode_image(current),
        }
        return jsonify(resp)

    except Exception as e:
        import traceback
        print('api_vision_pipeline error:', e)
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Template context processor để có thể sử dụng include trong template
@app.context_processor
def inject_template_vars():
    return dict()




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)