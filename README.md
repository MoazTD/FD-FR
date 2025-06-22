# Face Detection Project

A Python-based face detection and recognition system using OpenCV and machine learning techniques.
Check out this video demo: [My Awesome Demo Video]([https://www.youtube.com/watch?v=your-video-id](https://www.youtube.com/watch?v=j3ZrlK7Kn9Q))

## Project Overview

This project implements face detection and recognition capabilities with the following features:
- Real-time face detection from camera feed or images
- Face recognition using pre-trained models
- Support for known faces database
- YAML-based configuration management

## Files Structure

```
├── Project.py                          # Main project file
├── face_detection_1750591753.jpg       # Sample face detection image
├── face_detection_model.yaml           # Face detection model configuration
├── known_faces_opencv.pkl              # Serialized known faces data
├── label_encoder.yaml                  # Label encoding configuration
├── svm_face_model.joblib               # Trained SVM model for face recognition
└── README.md                           # This file
```

## Prerequisites

Before setting up the project, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)
- A webcam (for real-time face detection)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FD-FR
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv face_detection_env

# Activate virtual environment
# On Windows:
face_detection_env\Scripts\activate
# On macOS/Linux:
source face_detection_env/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install scikit-learn
pip install joblib
pip install PyYAML
pip install matplotlib
pip install pillow
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

### 4. Verify OpenCV Installation

Test if OpenCV is properly installed:

```python
import cv2
print(cv2.__version__)
```

## Configuration

### Model Files

The project uses several pre-trained models and configuration files:

- `face_detection_model.yaml`: Contains face detection model parameters
- `svm_face_model.joblib`: Trained SVM classifier for face recognition
- `known_faces_opencv.pkl`: Database of known faces encodings
- `label_encoder.yaml`: Label encoding mappings

### Adding Known Faces

To add new faces to the recognition system:

1. Place face images in a designated folder
2. Run the face encoding script to update `known_faces_opencv.pkl`
3. Update the label encoder accordingly

## Usage

### Running the Main Application

```bash
python Project.py
```

### Basic Usage Examples

```python
import cv2
from your_face_detection_module import FaceDetector

# Initialize face detector
detector = FaceDetector()

# For webcam feed
detector.detect_from_webcam()

# For image file
detector.detect_from_image("path/to/image.jpg")
```

## Features

### Face Detection
- Real-time face detection using OpenCV's DNN module or Haar cascades
- Bounding box visualization around detected faces
- Confidence score display

### Face Recognition
- Recognition of known faces using SVM classifier
- Support for multiple face encodings per person
- Configurable confidence thresholds

### Configuration Options
- Adjustable detection sensitivity
- Customizable output formats
- Flexible model parameters

## Troubleshooting

### Common Issues

1. **Camera not detected**
   ```
   Solution: Check camera permissions and ensure no other application is using the camera
   ```

2. **Import errors for OpenCV**
   ```bash
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-contrib-python
   ```

3. **Model files not found**
   ```
   Ensure all .yaml, .pkl, and .joblib files are in the project directory
   ```

4. **Low detection accuracy**
   ```
   Adjust confidence thresholds in configuration files
   Ensure good lighting conditions
   Check camera resolution settings
   ```

## Performance Optimization

- Use GPU acceleration if available (OpenCV with CUDA support)
- Optimize image resolution for better performance
- Adjust detection frequency for real-time applications
- Consider using threading for concurrent processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Requirements

Create a `requirements.txt` file with:
```
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0
joblib>=1.0.0
PyYAML>=5.4.0
matplotlib>=3.3.0
Pillow>=8.0.0
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Create an issue in the GitHub repository
- Review OpenCV documentation for advanced configurations

## Acknowledgments

- OpenCV community for computer vision tools
- scikit-learn for machine learning algorithms
- Contributors to face recognition research
