# MLDL Pothole Detection

A Flask web application for detecting potholes in images using YOLOv8 deep learning model.

## Project Overview

This project provides a web-based interface for uploading images and detecting potholes using a trained YOLOv8 model. The application uses Flask for the backend, OpenCV for image processing, and Ultralytics YOLOv8 for object detection.

## Project Structure

```
MLDL Potehole/
├── app.py                      # Flask web application
├── potholedetection.py         # Standalone detection script
├── requirements.txt            # Python dependencies
├── run_app.bat                 # Windows batch file to run the app
├── PotholeDetection.ipynb      # Jupyter notebook for model training
├── .gitignore                  # Git ignore file
├── static/
│   └── style.css               # CSS styles for the web interface
├── templates/
│   └── index.html              # HTML template for the web interface
├── uploads/                    # Uploaded images (generated at runtime)
├── results/                    # Detection results (generated at runtime)
├── pothole_yolo/               # Trained YOLOv8 model
│   └── pothole_yolo/
│       └── train1/
│           └── weights/
│               └── best.pt     # Trained model weights
└── dataset/                    # Training dataset (user-provided)
    └── dataset_Yolo/           # YOLO format dataset (user-provided)
```

## Prerequisites

- Python 3.8 or higher
- Anaconda/Miniconda (recommended)
- CUDA-capable GPU (optional, for faster inference)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "MLDL Potehole"
```

### 2. Create a Conda Environment (Recommended)

```bash
conda create -n MLDL python=3.10
conda activate MLDL
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install flask>=2.3.0
pip install ultralytics>=8.0.0
pip install numpy>=1.24.0
pip install opencv-python>=4.8.0
pip install pillow>=10.0.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
```

### 4. Prepare the Model

Place your trained YOLOv8 model weights in:
```
pothole_yolo/pothole_yolo/train1/weights/best.pt
```

The model should be trained using Ultralytics YOLOv8 format.

### 5. Prepare the Dataset (For Training)

If you want to retrain the model, place your dataset in:
- `dataset/` - Original images and annotations
- `dataset_Yolo/` - YOLO format annotations (train/val splits)

## Running the Application

### Using the Batch File (Windows)

Simply double-click `run_app.bat` or run it from the command line:

```cmd
run_app.bat
```

This will:
1. Activate the conda environment
2. Start the Flask server
3. Open your browser to http://127.0.0.1:5000

### Using Command Line

```bash
conda activate MLDL
python app.py
```

Then open your browser and navigate to: **http://127.0.0.1:5000**

## Usage

1. Open the web application in your browser
2. Click on "Choose File" to select an image
3. Click "Detect Potholes" to upload and process
4. View the results with bounding boxes around detected potholes

## Configuration

You can modify the following settings in [`app.py`](app.py):

| Setting | Description | Default |
|---------|-------------|---------|
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detections | 0.25 |
| `LINE_THICKNESS` | Bounding box line thickness | 4 |
| `MAX_CONTENT_LENGTH` | Maximum upload file size | 16MB |

## Model Training

To train your own YOLOv8 model:

1. Prepare your dataset in YOLO format
2. Use the [`PotholeDetection.ipynb`](PotholeDetection.ipynb) notebook
3. Or use the Ultralytics CLI:

```bash
yolo detect train data=<path-to-data.yaml> model=yolov8n.pt epochs=100
```

## Technologies Used

- **Flask** - Web framework
- **Ultralytics YOLOv8** - Object detection model
- **OpenCV** - Image processing
- **PyTorch** - Deep learning framework
- **Pillow** - Image handling

## License

This project is for educational and research purposes.

## Notes

- The `uploads/` and `results/` directories are automatically created at runtime
- Large files are excluded from version control (see [.gitignore](.gitignore))
- The model expects images in common formats: JPG, PNG, JPEG, GIF, BMP, WEBP
