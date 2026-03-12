"""
Flask Web Application for Pothole Detection using YOLOv8

This application provides a web interface for uploading images and videos
and detecting potholes using a trained YOLOv8 model.
Supports both batch processing and live real-time video streaming.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import threading
import time

# Configuration
app = Flask(__name__)

app.secret_key = 'pothole_detection_secret_key'

# Get the project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
MODEL_PATH = os.path.join(BASE_DIR, 'pothole_yolo', 'pothole_yolo', 'train1', 'weights', 'best.pt')

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Bounding box line thickness (higher = more visible)
LINE_THICKNESS = 4

# Blue color for bounding boxes (BGR format)
BOX_COLOR = (255, 0, 0)  # Blue

# Configure Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size for videos

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
    Render the main page with the upload form.
    
    Returns:
        Rendered HTML template
    """
    return render_template('index.html', live_mode=False)


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload, run inference, and display results.
    Supports both images and videos.
    
    Returns:
        Rendered HTML template with results or error
    """
    # Check if model is loaded
    if model is None:
        return render_template('index.html', error='Error: Model not loaded. Please check the model path.')
    
    # Check if file part exists
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected. Please choose a file.')
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return render_template('index.html', error='No file selected. Please choose a file.')
    
    # Check file type and process accordingly
    if allowed_video_file(file.filename):
        return process_video(file)
    elif allowed_image_file(file.filename):
        return process_image(file)
    else:
        return render_template('index.html', 
            error='Invalid file type. Please upload an image (jpg, jpeg, png, gif, bmp, webp) or video (mp4, avi, mov, mkv).')


@app.route('/live-video')
def live_video_page():
    """
    Render the live video streaming page.
    
    Returns:
        Rendered HTML template
    """
    return render_template('index.html', live_mode=True)


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
    
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    
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
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_video_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
    
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
        
        current_video_path = filename
        is_processing_live = True
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'message': 'Live video processing started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        
        # Count detections
        num_detections = 0
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                num_detections = len(result.boxes)
        
        # Prepare URLs for display
        original_url = url_for('uploaded_file', filename=f'uploads/{filename}')
        result_url = url_for('uploaded_file', filename=f'results/output/{result_filename}')
        
        return render_template(
            'index.html',
            original_image=original_url,
            result_image=result_url,
            num_detections=num_detections,
            filename=filename,
            is_video=False
        )
        
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {str(e)}')


def process_video(file):
    """
    Process an uploaded video and detect potholes in each frame.
    Draws blue bounding boxes around detected potholes.
    
    Args:
        file: The uploaded file object
        
    Returns:
        Rendered HTML template with video results
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
        
        # Save uploaded video
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Open video for processing
        cap = cv2.VideoCapture(upload_path)
        
        if not cap.isOpened():
            return render_template('index.html', error='Error: Could not open video file.')
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video settings
        output_filename = f"result_{os.path.splitext(filename)[0]}.mp4"
        output_path = os.path.join(RESULTS_FOLDER, 'output', output_filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(RESULTS_FOLDER, 'output'), exist_ok=True)
        
        # Define video codec and create VideoWriter
        # Try different codecs
        fourcc = None
        for codec in ['mp4v', 'avc1', 'H264', 'XVID']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
            except:
                continue
        
        if fourcc is None or not out.isOpened():
            # Fallback: use mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        frame_count = 0
        total_detections = 0
        
        while True:
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
            
            # Count detections in this frame
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    total_detections += len(result.boxes)
            
            # Draw blue bounding boxes on the frame
            frame_with_boxes = draw_detections(frame, results)
            
            # Write the processed frame to output video
            out.write(frame_with_boxes)
            
            frame_count += 1
        
        # Release video objects
        cap.release()
        out.release()
        
        # Prepare URLs for display
        original_url = url_for('uploaded_file', filename=f'uploads/{filename}')
        result_url = url_for('uploaded_file', filename=f'results/output/{output_filename}')
        
        return render_template(
            'index.html',
            original_video=original_url,
            result_video=result_url,
            num_detections=total_detections,
            filename=filename,
            is_video=True,
            video_fps=fps
        )
        
    except Exception as e:
        return render_template('index.html', error=f'Error processing video: {str(e)}')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Serve uploaded files from the uploads directory.
    
    Args:
        filename: Path to the file
        
    Returns:
        File response
    """
    return send_from_directory(BASE_DIR, filename)


@app.route('/results/<path:filename>')
def result_file(filename):
    """
    Serve result files from the results directory.
    
    Args:
        filename: Path to the file
        
    Returns:
        File response
    """
    return send_from_directory(RESULTS_FOLDER, filename)


@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Handle file size exceeded error.
    
    Returns:
        Error message for file too large
    """
    flash('File too large. Maximum size is 100MB for videos.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle internal server errors.
    
    Returns:
        Error message for internal server error
    """
    flash('Internal server error. Please try again.', 'error')
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
