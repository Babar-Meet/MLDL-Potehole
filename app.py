"""
Flask Web Application for Pothole Detection using YOLOv8

This application provides a web interface for uploading images and detecting
potholes using a trained YOLOv8 model.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Bounding box line thickness (higher = more visible)
LINE_THICKNESS = 4

# Configure Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_extension(filename):
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: File extension in lowercase
    """
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


@app.route('/')
def index():
    """
    Render the main page with the upload form.
    
    Returns:
        Rendered HTML template
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload, run inference, and display results.
    
    Returns:
        Redirect to index with results or error message
    """
    # Check if model is loaded
    if model is None:
        flash('Error: Model not loaded. Please check the model path.', 'error')
        return redirect(url_for('index'))
    
    # Check if file part exists
    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('index'))
    
    # Validate file
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image (jpg, jpeg, png, gif, bmp, webp).', 'error')
        return redirect(url_for('index'))
    
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
            line_width=LINE_THICKNESS,  # Thicker lines for better visibility
            show_labels=True,  # Show class labels
            show_conf=True  # Show confidence scores
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
        
        # Prepare URLs for display - use direct file paths
        # The uploads folder is served from the project root
        original_url = url_for('uploaded_file', filename=f'uploads/{filename}')
        result_url = url_for('uploaded_file', filename=f'results/output/{result_filename}')
        
        return render_template(
            'index.html',
            original_image=original_url,
            result_image=result_url,
            num_detections=num_detections,
            filename=filename
        )
        
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


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
    flash('File too large. Maximum size is 16MB.', 'error')
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


if __name__ == '__main__':
    """
    Run the Flask application.
    """
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"Model path: {MODEL_PATH}")
    print("\nStarting Flask server...")
    print("Go to http://127.0.0.1:5000 to use the application")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
