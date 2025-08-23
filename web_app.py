#!/usr/bin/env python3
"""
TrafficAI Web Application
Modern web interface for helmet detection and license plate extraction
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import json
import logging
from datetime import datetime
import uuid

# Import our custom modules
from app.helmet_detector import HelmetDetector
from app.license_plate_detector import LicensePlateDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size for videos
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', 'm4v'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize models
helmet_detector = None
license_plate_detector = None

def initialize_models():
    """Initialize both detection models."""
    global helmet_detector, license_plate_detector
    
    try:
        helmet_detector = HelmetDetector(
            model_path="models/helmet_yolov8_gpu.pt",
            confidence=0.5
        )
        print("✅ Helmet detector initialized")
    except Exception as e:
        print(f"❌ Error initializing helmet detector: {e}")
        helmet_detector = None
    
    try:
        license_plate_detector = LicensePlateDetector(
            model_path="models/license_plate_yolov8.pt",
            confidence_threshold=0.5
        )
        print("✅ License plate detector initialized")
    except Exception as e:
        print(f"❌ Error initializing license plate detector: {e}")
        license_plate_detector = None

def make_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def process_video_frames(filepath, process_helmet, process_license):
    """Process video frames and extract detection results."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Could not read video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Processing video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    # Extract frames at 1 FPS (1 frame per second)
    frame_interval = int(fps) if fps > 0 else 30
    frame_results = []
    annotated_frames = []
    
    # Overall statistics
    total_helmet_detections = 0
    total_with_helmet = 0
    total_without_helmet = 0
    total_license_plates = 0
    
    frame_count = 0
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame at intervals
            if frame_count % frame_interval == 0:
                processed_frames += 1
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                print(f"Processing frame {frame_count} (t={timestamp:.2f}s)")
                
                frame_result = {
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'second': int(timestamp),
                    'helmet_results': None,
                    'license_plate_results': None,
                    'image': frame,  # Keep for annotation
                    'frame_base64': None,
                    'safety_status': 'warning'  # safe, unsafe, warning
                }
                
                # Save frame temporarily for processing
                temp_frame_path = f"temp_frame_{frame_count}.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    # Process with helmet detector
                    if process_helmet and helmet_detector:
                        helmet_results = helmet_detector.process_image(temp_frame_path, save_result=False)
                        if helmet_results:
                            cleaned_results = make_json_serializable(helmet_results)
                            cleaned_results['with_helmet_count'] = cleaned_results.get('helmet_count', 0)
                            frame_result['helmet_results'] = cleaned_results
                            
                            # Determine safety status based on helmet detection
                            with_helmet = cleaned_results.get('with_helmet_count', 0)
                            without_helmet = cleaned_results.get('no_helmet_count', 0)
                            
                            if without_helmet > 0:
                                frame_result['safety_status'] = 'unsafe'
                            elif with_helmet > 0:
                                frame_result['safety_status'] = 'safe'
                            else:
                                frame_result['safety_status'] = 'warning'
                            
                            # Update totals
                            total_helmet_detections += cleaned_results.get('total_detections', 0)
                            total_with_helmet += cleaned_results.get('with_helmet_count', 0)
                            total_without_helmet += cleaned_results.get('no_helmet_count', 0)
                    
                    # Process with license plate detector
                    if process_license and license_plate_detector:
                        try:
                            license_results = license_plate_detector.detect_plates(frame)
                            if license_results and len(license_results) > 1:
                                plates = make_json_serializable(license_results[1]) if license_results[1] else []
                                frame_result['license_plate_results'] = {
                                    'detections': len(plates),
                                    'details': plates
                                }
                                total_license_plates += len(plates)
                                
                                # Use the annotated image from license plate detection
                                if license_results[0] is not None:
                                    annotated_frame = license_results[0]
                            else:
                                frame_result['license_plate_results'] = {'detections': 0, 'details': []}
                        except Exception as e:
                            print(f"License plate detection error on frame {frame_count}: {e}")
                            frame_result['license_plate_results'] = {'error': str(e)}
                    
                    # Draw bounding boxes on frame before converting to base64
                    annotated_frame = frame.copy()
                    if process_helmet and helmet_detector and frame_result['helmet_results']:
                        try:
                            detections = frame_result['helmet_results'].get('detections', [])
                            if detections:
                                annotated_frame = helmet_detector.annotate_image(annotated_frame, detections, None)
                        except Exception as e:
                            print(f"❌ Error annotating frame {frame_count}: {e}")
                    
                    # Convert annotated frame to base64 for frontend display
                    try:
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        frame_result['frame_base64'] = f"data:image/jpeg;base64,{frame_base64}"
                        print(f"✅ Frame {frame_count}: Generated annotated base64 image ({len(frame_base64)} chars)")
                    except Exception as e:
                        print(f"❌ Error encoding frame {frame_count}: {e}")
                        frame_result['frame_base64'] = None
                    
                    # Store frame temporarily for video timeline but remove image array before adding to results
                    frame_result_for_storage = frame_result.copy()
                    # Safely remove image key if it exists
                    frame_result_for_storage.pop('image', None)
                    
                    frame_results.append(frame_result_for_storage)
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
            
            frame_count += 1
            
            # Limit processing for very long videos (max 100 frames)
            if processed_frames >= 100:
                print(f"Reached maximum frames limit (100), stopping processing")
                break
                
    finally:
        cap.release()
    
    # Clean up frame_results to ensure JSON serialization
    clean_frame_results = []
    for frame in frame_results:
        clean_frame = make_json_serializable(frame)
        # Safely remove image key if it exists
        clean_frame.pop('image', None)
        clean_frame_results.append(clean_frame)
    
    return {
        'file_type': 'video',  # Add file_type to video results
        'video_results': {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': float(fps),
            'duration': float(duration),
            'frame_interval': frame_interval
        },
        'helmet_results': {
            'total_detections': total_helmet_detections,
            'with_helmet_count': total_with_helmet,
            'no_helmet_count': total_without_helmet
        },
        'license_plate_results': {
            'detections': total_license_plates,
            'details': []  # Individual frame details available in frame_results
        },
        'frames': clean_frame_results,
        'video_timeline': clean_frame_results  # For frontend timeline display
    }

# Initialize models on startup
initialize_models()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported formats: PNG, JPG, GIF, MP4, AVI, MOV, WMV, FLV, WEBM'}), 400
        
        # Get processing options
        process_helmet = request.form.get('process_helmet', 'true').lower() == 'true'
        process_license = request.form.get('process_license', 'true').lower() == 'true'
        
        # Ensure at least one detection is enabled, prefer enabling both
        if not process_helmet and not process_license:
            process_helmet = True
            process_license = True
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        saved_filename = f"{timestamp}_{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Process the file
        results = process_file(filepath, process_helmet, process_license)
        
        # Debug logging
        print(f"📤 Returning results: file_type={results.get('file_type')}")
        if results.get('video_results'):
            vr = results['video_results']
            print(f"📹 Video stats: {vr['processed_frames']} frames, {vr['duration']:.1f}s, {vr['fps']:.1f} FPS")
        if results.get('video_timeline'):
            print(f"🎬 Timeline: {len(results['video_timeline'])} frame entries")
            # Check if first frame has base64 data
            if len(results['video_timeline']) > 0:
                first_frame = results['video_timeline'][0]
                has_base64 = 'frame_base64' in first_frame and first_frame['frame_base64'] is not None
                print(f"🖼️  First frame has base64: {has_base64}")
                if has_base64:
                    print(f"📏 Base64 length: {len(first_frame['frame_base64'])} chars")
        
        return jsonify(results)
        
    except Exception as e:
        logging.error(f"Error processing upload: {e}")
        return jsonify({'error': str(e)}), 500

def process_file(filepath, process_helmet, process_license):
    """Process uploaded file with selected detectors."""
    results = {
        'filename': os.path.basename(filepath),
        'helmet_results': None,
        'license_plate_results': None,
        'annotated_image': None,
        'file_type': 'image',
        'video_results': None
    }
    
    # Check if file is video or image
    file_ext = filepath.lower().split('.')[-1]
    video_extensions = ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', 'm4v']
    is_video = file_ext in video_extensions
    
    if is_video:
        # Process video frames
        video_results = process_video_frames(filepath, process_helmet, process_license)
        results.update(video_results)
        
        # For videos, we don't need to process again as frames are already processed
        return results
    else:
        # Read image
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Could not read image file")
    
    # Process with helmet detector
    if process_helmet and helmet_detector:
        try:
            helmet_results = helmet_detector.process_image(filepath, save_result=False)
            # Convert to JSON-serializable format
            if helmet_results:
                cleaned_results = make_json_serializable(helmet_results)
                # Add correct field names for frontend compatibility
                cleaned_results['with_helmet_count'] = cleaned_results.get('helmet_count', 0)
                results['helmet_results'] = cleaned_results
        except Exception as e:
            logging.error(f"Helmet detection error: {e}")
            results['helmet_results'] = {'error': str(e)}
    
    # Process with license plate detector
    if process_license and license_plate_detector:
        try:
            license_results = license_plate_detector.detect_plates(image)
            detections = []
            if license_results and len(license_results) > 1 and license_results[1]:
                # Convert detection results to JSON-serializable format
                raw_detections = license_results[1]
                if isinstance(raw_detections, list):
                    detections = make_json_serializable(raw_detections)
                else:
                    detections = []
            
            results['license_plate_results'] = {
                'detections': len(detections) if isinstance(detections, list) else 0,
                'details': detections if isinstance(detections, list) else []
            }
        except Exception as e:
            logging.error(f"License plate detection error: {e}")
            results['license_plate_results'] = {'error': str(e), 'detections': 0, 'details': []}
    
    # Create annotated image - start with license plate annotations if available
    if process_license and license_plate_detector and results['license_plate_results'] and not results['license_plate_results'].get('error'):
        # Use the annotated image from license plate detection
        annotated_image = license_results[0] if license_results and license_results[0] is not None else image.copy()
    else:
        annotated_image = image.copy()
    
    # Draw helmet detection results
    if process_helmet and helmet_detector and results['helmet_results'] and not results['helmet_results'].get('error'):
        try:
            # Use helmet detector's annotation method with the image, not filepath
            detections = results['helmet_results'].get('detections', [])
            if detections and hasattr(helmet_detector, 'annotate_image'):
                # Pass the actual image array instead of filepath
                annotated_image = helmet_detector.annotate_image(annotated_image, detections, None)
        except Exception as e:
            logging.error(f"Error drawing helmet results: {e}")
    
    # Save annotated image with proper extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    annotated_filename = f"annotated_{base_name}.jpg"
    annotated_path = os.path.join(app.config['RESULTS_FOLDER'], annotated_filename)
    cv2.imwrite(annotated_path, annotated_image)
    
    # Convert to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', annotated_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    results['annotated_image'] = f"data:image/jpeg;base64,{img_base64}"
    
    return results

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'helmet_detector': helmet_detector is not None,
        'license_plate_detector': license_plate_detector is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting TrafficAI Web Application...")
    print("📱 Open your browser and go to: http://localhost:3000")
    print("🔍 Health check: http://localhost:3000/health")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=3000)
