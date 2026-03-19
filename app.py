"""
Flask Web Application for Pothole Detection using YOLOv8

This application provides a web interface for uploading images
and detecting potholes using a trained YOLOv8 model.
Supports live real-time video streaming.
"""

import os
import cv2
import uuid
import json
import hashlib
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory, Response, render_template, flash, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import threading
import time
import queue

# Configuration
app = Flask(__name__)
CORS(app)

# Secret key (override in production via SECRET_KEY env var)
app.secret_key = os.getenv('SECRET_KEY', 'pothole_detection_secret_key')

# Get the project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (can be overridden via env vars for deployment)
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', os.path.join(BASE_DIR, 'results'))
MODEL_PATH = os.getenv(
    'MODEL_PATH',
    os.path.join(BASE_DIR, 'pothole_yolo', 'pothole_yolo', 'train1', 'weights', 'best.pt')
)

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Bounding box line thickness (higher = more visible)
LINE_THICKNESS = 3

# Blue color for bounding boxes (BGR format)
BOX_COLOR = (255, 0, 0)  # Blue

# Build timestamp - captures when the application was deployed/started
BUILD_TIMESTAMP = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
BUILD_TIMESTAMP_ISO = datetime.now(timezone.utc).isoformat()

# Configure Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size for videos
app.config['CHUNK_SIZE'] = 5 * 1024 * 1024  # 5MB chunk size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLOv8 model
print(f"Loading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Global variable for live video processing
current_video_path = None
is_processing_live = False
live_video_lock = threading.Lock()


def allowed_file(filename, allowed_extensions):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        allowed_extensions: Set of allowed extensions
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def allowed_image_file(filename):
    """Check if the file is an allowed image type."""
    return allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS)


def allowed_video_file(filename):
    """Check if the file is an allowed video type."""
    return allowed_file(filename, ALLOWED_VIDEO_EXTENSIONS)


def get_file_extension(filename):
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: File extension in lowercase
    """
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def draw_detections(frame, results):
    """
    Draw blue bounding boxes around detected potholes on the frame.
    
    Args:
        frame: numpy array - the video frame
        results: YOLO results object
        
    Returns:
        numpy array - frame with drawn bounding boxes
    """
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence score
                conf = float(box.conf[0])
                
                # Draw blue rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, LINE_THICKNESS)
                
                # Draw label background
                label = f'Pothole {conf:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), BOX_COLOR, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


@app.route('/')
def index():
    """
    Render the main page.
    
    Returns:
        Rendered HTML template
    """
    return render_template('index.html', live_mode=False, result_image=None, original_image=None, filename=None, result_video=None, error=None, processing=False, num_detections=0, build_timestamp=BUILD_TIMESTAMP, build_timestamp_iso=BUILD_TIMESTAMP_ISO)

@app.route('/health')
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON status message
    """
    return jsonify({"status": "healthy"})


@app.route('/build-info')
def build_info():
    """
    Return build/deployment timestamp information.
    
    Returns:
        JSON response with build timestamp
    """
    return jsonify({
        "build_timestamp": BUILD_TIMESTAMP,
        "build_timestamp_iso": BUILD_TIMESTAMP_ISO,
        "message": "Application build/deployment timestamp"
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload, run inference, and display results.
    Supports images only. Videos are handled via live video streaming.
    
    Returns:
        JSON response with results or error
    """
    # Check if model is loaded
    if model is None:
        return render_template('index.html', 
            live_mode=False,
            result_image=None,
            original_image=None,
            filename=None,
            result_video=None,
            error='Error: Model not loaded. Please check the model path.',
            processing=False,
            num_detections=0,
            build_timestamp=BUILD_TIMESTAMP,
            build_timestamp_iso=BUILD_TIMESTAMP_ISO
        ), 500
    
    # Check if file part exists
    if 'file' not in request.files:
        return render_template('index.html', 
            live_mode=False,
            result_image=None,
            original_image=None,
            filename=None,
            result_video=None,
            error='No file selected. Please choose a file.',
            processing=False,
            num_detections=0,
            build_timestamp=BUILD_TIMESTAMP,
            build_timestamp_iso=BUILD_TIMESTAMP_ISO
        ), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return render_template('index.html', 
            live_mode=False,
            result_image=None,
            original_image=None,
            filename=None,
            result_video=None,
            error='No file selected. Please choose a file.',
            processing=False,
            num_detections=0,
            build_timestamp=BUILD_TIMESTAMP,
            build_timestamp_iso=BUILD_TIMESTAMP_ISO
        ), 400
    
    # Check file type and process accordingly
    if allowed_image_file(file.filename):
        return process_image(file)
    else:
        return render_template('index.html', 
            live_mode=False,
            result_image=None,
            original_image=None,
            filename=None,
            result_video=None,
            error='Invalid file type. Please upload an image (jpg, jpeg, png, gif, bmp, webp).',
            processing=False,
            num_detections=0,
            build_timestamp=BUILD_TIMESTAMP,
            build_timestamp_iso=BUILD_TIMESTAMP_ISO
        ), 400


@app.route('/live-video')
def live_video_page():
    """
    Redirect to frontend for live video functionality.
    
    Returns:
        JSON status
    """
    return jsonify({"error": "This is an API only. Live video UI is handled by frontend."}), 400


# ==================== Live Video Streaming ====================


def process_video_generator(video_path):
    """
    Process video frame by frame and yield processed frames for streaming.
    
    Args:
        video_path: Path to the video file
        
    Yields:
        JPEG frames with detection boxes
    """
    global is_processing_live
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        yield None
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    frame_time = 1.0 / fps
    
    try:
        while is_processing_live:
            ret, frame = cap.read()
            if not ret:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Run YOLOv8 inference on this frame
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                save=False,
                verbose=False
            )
            
            # Draw bounding boxes on the frame
            frame_with_boxes = draw_detections(frame, results)
            
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame_with_boxes)
            if not ret:
                continue
                
            frame_bytes = jpeg.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control frame rate
            time.sleep(frame_time)
            
    finally:
        cap.release()
        is_processing_live = False


@app.route('/video_feed/<path:filename>')
def video_feed(filename):
    """
    Stream processed video frames in real-time.
    
    Args:
        filename: The video filename to process
        
    Returns:
        Streaming response with JPEG frames
    """
    global is_processing_live
    
    # Stop any existing processing
    is_processing_live = False
    time.sleep(0.5)
    
    # Secure the filename to prevent path traversal
    safe_filename = secure_filename(os.path.basename(filename))
    video_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    
    if not os.path.exists(video_path):
        return "Video not found", 404
    
    is_processing_live = True
    
    return Response(
        process_video_generator(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start_live_video', methods=['POST'])
def start_live_video():
    """
    Start live video processing from uploaded file.
    
    Returns:
        JSON response with status
    """
    global current_video_path, is_processing_live
    
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file selected',
            'progress': 0
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected',
            'progress': 0
        }), 400
    
    if not allowed_video_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type. Please upload a video file.',
            'progress': 0
        }), 400
    
    try:
        # Stop any existing processing
        is_processing_live = False
        time.sleep(0.5)
        
        # Save the video
        original_filename = secure_filename(file.filename)
        filename = original_filename
        counter = 1
        while os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            counter += 1
        
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        file.flush()
        os.fsync(file.fileno())  # Force write to disk
        
        # Verify file exists and has size
        if not (os.path.exists(video_path) and os.path.getsize(video_path) > 0):
            return jsonify({
                'status': 'error',
                'message': 'Error saving video file',
                'progress': 0
            }), 500
        
        current_video_path = filename
        is_processing_live = True
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'message': 'Live video processing started',
            'progress': 100
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing video: {str(e)}',
            'progress': 0
        }), 500


@app.route('/stop_live_video', methods=['POST'])
def stop_live_video():
    """
    Stop live video processing.
    
    Returns:
        JSON response with status
    """
    global is_processing_live
    is_processing_live = False
    return jsonify({'status': 'success', 'message': 'Live video processing stopped'})


def process_image(file):
    """
    Process an uploaded image and detect potholes.
    
    Args:
        file: The uploaded file object
        
    Returns:
        Rendered HTML template with results
    """
    try:
        # Secure the filename and create unique name
        original_filename = secure_filename(file.filename)
        filename = original_filename
        counter = 1
        while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            counter += 1
        
        # Save uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Run inference with visible bounding boxes
        results = model.predict(
            source=upload_path,
            conf=CONFIDENCE_THRESHOLD,
            save=True,
            project=RESULTS_FOLDER,
            name='output',
            exist_ok=True,
            line_width=LINE_THICKNESS,
            show_labels=True,
            show_conf=True
        )
        
        # Get the result image path
        result_filename = f"{os.path.splitext(filename)[0]}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, 'output', result_filename)
        
        # If result doesn't exist, try with the original extension
        if not os.path.exists(result_path):
            result_filename = filename
            result_path = os.path.join(RESULTS_FOLDER, 'output', result_filename)
        
        # If still not found, look for any jpg/png in the output folder
        if not os.path.exists(result_path):
            output_dir = os.path.join(RESULTS_FOLDER, 'output')
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        result_filename = f
                        result_path = os.path.join(output_dir, f)
                        break
        
        
        # Prepare URLs for display (these would be URLs relative to the API host)
        original_url = f'/uploads/{filename}'
        result_url = f'/results/output/{result_filename}'
        
        # Count detections
        num_detections = 0
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                num_detections = len(result.boxes)
        
        return render_template('index.html', 
            live_mode=False,
            result_image=result_url,
            original_image=original_url,
            filename=filename,
            result_video=None,
            error=None,
            processing=False,
            num_detections=num_detections,
            build_timestamp=BUILD_TIMESTAMP,
            build_timestamp_iso=BUILD_TIMESTAMP_ISO
        )
        
    except Exception as e:
        return render_template('index.html', 
            live_mode=False,
            result_image=None,
            original_image=None,
            filename=None,
            result_video=None,
            error=f'Error processing image: {str(e)}',
            processing=False,
            num_detections=0,
            build_timestamp=BUILD_TIMESTAMP,
            build_timestamp_iso=BUILD_TIMESTAMP_ISO
        )


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Serve uploaded files from the uploads directory.
    
    Args:
        filename: Path to the file
        
    Returns:
        File response
    """
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/results/<path:filename>')
def result_file(filename):
    """
    Serve result files from the results directory.
    
    Args:
        filename: Path to the file
        
    Returns:
        File response
    """
    # Handle nested paths for results/output/filename
    if filename.startswith('output/'):
        return send_from_directory(os.path.join(RESULTS_FOLDER, 'output'), filename.replace('output/', ''))
    return send_from_directory(RESULTS_FOLDER, filename)


@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Handle file size exceeded error.
    
    Returns:
        Error message for file too large
    """
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 100MB for videos.',
        'error': 'FileTooLarge'
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle internal server errors.
    
    Returns:
        Error message for internal server error
    """
    return jsonify({
        'status': 'error',
        'message': 'Internal server error. Please try again.',
        'error': 'ServerError'
    }), 500


@app.errorhandler(404)
def not_found(error):
    """
    Handle not found errors.
    
    Returns:
        Error message for not found
    """
    return jsonify({
        'status': 'error',
        'message': 'Resource not found.',
        'error': 'NotFound'
    }), 404


if __name__ == "__main__":
    # Support PORT env var for local container runs
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
