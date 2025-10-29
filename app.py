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
    from vision_core.segmentation import (
        global_thresholding,
        Thresholding,
        adaptiveThreshold,
        otsu,
        regionGrowing,
        watershed_segment,
        detect_by_connected_components,
        detect_by_contours,
        compute_chain_code,
        draw_chain_code,
        compute_iou,
    )
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

# Segmentation page
@app.route('/segmentation/')
def segmentation_page():
    # Provide available ops to template
    methods = [
        {'value': 'global_threshold', 'label': 'Global Thresholding'},
        {'value': 'adaptive_threshold', 'label': 'Adaptive Thresholding'},
        {'value': 'otsu', 'label': "Otsu's Method"},
        {'value': 'region_growing', 'label': 'Region Growing'},
        {'value': 'watershed', 'label': 'Watershed'},
        {'value': 'connected_components', 'label': 'Connected Components'},
        {'value': 'contours', 'label': 'Contour Detection'},
        {'value': 'chain_code', 'label': 'Chain Code'},
    ]
    return render_template('segmentation.html', methods=methods)

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

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError('None image')
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f'Unsupported image shape: {img.shape}')

@app.route('/api/segmentation/visualize-chain-code', methods=['POST'])
def api_visualize_chain_code():
    try:
        data = request.get_json(silent=True) or {}
        chain_code_str = (data.get('chain_code') or '').strip()
        # If fit is True (default), auto-resize to fit canvas; otherwise use start_point + scale
        fit = data.get('fit') if data.get('fit') is not None else True
        start_point = data.get('start_point') or [50, 50]
        scale = int(data.get('scale') or 8)

        # sanitize chain code: keep only digits 0-7
        chain_code_str = ''.join([ch for ch in chain_code_str if ch in '01234567'])
        if not chain_code_str:
            return jsonify({'success': False, 'error': 'Empty chain code'}), 400

        # directions (dy, dx) mapping consistent with segmentation.py
        directions = [
            (0, 1),   # 0: right
            (-1, 1),  # 1: up-right
            (-1, 0),  # 2: up
            (-1, -1), # 3: up-left
            (0, -1),  # 4: left
            (1, -1),  # 5: down-left
            (1, 0),   # 6: down
            (1, 1)    # 7: down-right
        ]

        # Build list of codes and delegate rendering to segmentation.draw_chain_code
        codes = [ord(ch) - 48 for ch in chain_code_str if ch in '01234567']
        # draw_chain_code from segmentation returns a BGR numpy image rendered via matplotlib
        img = draw_chain_code(codes)
        return jsonify({'success': True, 'image': _encode_image(img)})
    except Exception as e:
        import traceback
        print('api_visualize_chain_code error:', e)
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/api/segmentation/process', methods=['POST'])
def api_segmentation_process():
    try:
        # Read image from multipart or JSON
        img = None
        params = {}
        op = None

        if 'image' in request.files:
            file = request.files['image']
            if not file or file.filename == '':
                return jsonify({'error': 'No file provided'}), 400
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            op = (request.form.get('op') or '').lower()
            params = {k: v for k, v in request.form.items() if k != 'op'}
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get('image_data')
            op = (data.get('op') or '').lower()
            params = data.get('params') or {}
            if b64:
                if b64.startswith('data:image'):
                    b64 = b64.split(',')[1]
                image_bytes = base64.b64decode(b64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'No image provided or decode failed'}), 400

        gray = _ensure_gray(img)

        # Dispatch operations
        op = (op or 'global_threshold').lower()
        resp = {'success': True, 'op': op}

        if op == 'global_threshold':
            eps = float(params.get('epsilon', 1)) if params else 1
            T = global_thresholding(gray, epsilon=eps)
            out = Thresholding(gray, T=int(T))
            resp['threshold'] = int(T)
            resp['output'] = _encode_image(out)

        elif op == 'adaptive_threshold':
            block_size = int(params.get('block_size', 15)) if params else 15
            # ensure odd and >=3
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1
            C = int(params.get('C', 5)) if params else 5
            out = adaptiveThreshold(gray, block_size=block_size, C=C)
            resp['params'] = {'block_size': block_size, 'C': C}
            resp['output'] = _encode_image(out)

        elif op == 'otsu':
            T = int(otsu(gray))
            out = Thresholding(gray, T=T)
            resp['threshold'] = T
            resp['output'] = _encode_image(out)

        elif op == 'region_growing':
            # seed (x,y), T - with auto T calculation options
            h, w = gray.shape
            seed_x = int(params.get('seed_x', w // 2)) if params else w // 2
            seed_y = int(params.get('seed_y', h // 2)) if params else h // 2
            
            # T method: 'manual', 'global', or 'otsu'
            t_method = (params.get('t_method') or 'manual').lower()
            if t_method == 'global':
                th = int(global_thresholding(gray, epsilon=1))
            elif t_method == 'otsu':
                th = int(otsu(gray))
            else:  # manual
                th = int(params.get('T', 10)) if params else 10
            
            out = regionGrowing(gray, seed=(seed_x, seed_y), T=th)
            resp['params'] = {'seed_x': seed_x, 'seed_y': seed_y, 'T': th, 't_method': t_method}
            resp['threshold'] = th
            resp['output'] = _encode_image(out)

        elif op == 'watershed':
            blur_ksize = int(params.get('blur_ksize', 3)) if params else 3
            dist_ratio = float(params.get('dist_ratio', 0.1)) if params else 0.1
            markers_ws, result_img, boundary_mask = watershed_segment(gray, blur_ksize=blur_ksize, dist_ratio=dist_ratio)
            
            # Generate additional outputs for visualization
            # Sure foreground (markers > 1)
            sure_fg = np.zeros_like(gray)
            sure_fg[markers_ws > 1] = 255
            
            # Sure background (markers == 1)
            sure_bg = np.zeros_like(gray)
            sure_bg[markers_ws == 1] = 255
            
            # Unknown region (markers == 0, but not boundary)
            unknown = np.zeros_like(gray)
            unknown[(markers_ws == 0) & (~boundary_mask)] = 255
            
            # Color visualizations (Red channel emphasized)
            sure_fg_color = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)
            sure_fg_color[:, :, 2] = sure_fg  # Red channel
            
            sure_bg_color = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)
            sure_bg_color[:, :, 0] = sure_bg  # Blue channel
            
            unknown_color = cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
            unknown_color[:, :, 1] = unknown  # Green channel
            
            resp['params'] = {'blur_ksize': blur_ksize, 'dist_ratio': dist_ratio}
            resp['output'] = _encode_image(result_img)
            resp['sure_foreground'] = _encode_image(sure_fg_color)
            resp['sure_background'] = _encode_image(sure_bg_color)
            resp['unknown_region'] = _encode_image(unknown_color)

        elif op == 'connected_components':
            count, label_img = detect_by_connected_components(gray)
            resp['count'] = int(count)
            resp['output'] = _encode_image(label_img)

        elif op == 'contours':
            count, contour_img, _ = detect_by_contours(gray)
            resp['count'] = int(count)
            resp['output'] = _encode_image(contour_img)

        elif op == 'chain_code':
            # First threshold the image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try:
                chain_code, cnt = compute_chain_code(binary)
                # Draw the contour on the image
                result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                
                # Create chain code string (limit to first 100 for display)
                chain_str = ''.join(map(str, chain_code[:100]))
                if len(chain_code) > 100:
                    chain_str += f'... ({len(chain_code)} total)'
                
                resp['chain_code'] = chain_str
                resp['chain_length'] = len(chain_code)
                # Also include the full chain code for visualization
                resp['chain_code_raw'] = ''.join(map(str, chain_code))
                resp['output'] = _encode_image(result)
            except Exception as e:
                return jsonify({'error': f'Chain code error: {str(e)}'}), 400

        else:
            return jsonify({'error': f'Unsupported operation: {op}'}), 400
        
        # Handle Ground Truth IOU if provided
        if 'ground_truth' in request.files:
            gt_file = request.files['ground_truth']
            if gt_file and gt_file.filename:
                gt_nparr = np.frombuffer(gt_file.read(), np.uint8)
                gt_img = cv2.imdecode(gt_nparr, cv2.IMREAD_GRAYSCALE)
                if gt_img is not None and 'output' in resp:
                    # Decode the output image from base64 to compute IOU
                    # For now, use the binary result directly if available
                    # We need to get the actual mask from the operation
                    result_mask = None
                    if op in ['global_threshold', 'adaptive_threshold', 'otsu', 'region_growing']:
                        # These operations return binary masks
                        result_mask = out
                    elif op == 'connected_components':
                        result_mask = (label_img > 0).astype(np.uint8) * 255
                    elif op == 'contours':
                        result_mask = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
                        _, result_mask = cv2.threshold(result_mask, 1, 255, cv2.THRESH_BINARY)
                    
                    if result_mask is not None:
                        # Resize gt to match result if needed
                        if gt_img.shape != result_mask.shape:
                            gt_img = cv2.resize(gt_img, (result_mask.shape[1], result_mask.shape[0]))
                        iou_score = compute_iou(result_mask, gt_img)
                        resp['iou'] = float(iou_score)
                        resp['ground_truth'] = _encode_image(gt_img)

        return jsonify(resp)

    except Exception as e:
        import traceback
        print('api_segmentation_process error:', e)
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

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