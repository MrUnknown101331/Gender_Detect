# Real-Time Gender Detection with Deep Learning

## Introduction

This project implements a real-time gender detection system using deep learning and TensorFlow/Keras. It uses a pre-trained Convolutional Neural Network (CNN) to analyze facial features from a live video stream (webcam) and predict the gender (Male or Female) of detected faces.

## Features

*   Real-time gender prediction from webcam feed.
*   Face detection using MediaPipe.
*   Clear visualization with bounding boxes and gender labels.
*   Model training script (provided) for creating or retraining the gender classification model.

## Dependencies

*   Python 3.x
*   TensorFlow/Keras
*   OpenCV (cv2)
*   NumPy
*   MediaPipe
*   scikit-learn
*   matplotlib
*   pandas

You can install these using pip:

```bash
pip install tensorflow opencv-python numpy mediapipe scikit-learn matplotlib pandas
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mrunknown101331/Gender_Detect.git
cd Gender_Detect
```

(Optional but recommended) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # Activate on Linux/macOS
venv\Scripts\activate.bat  # Activate on Windows
```

Install the required libraries (using requirements.txt is recommended):

```bash
pip install -r requirements.txt # If you have a requirements.txt file
# OR
pip install tensorflow opencv-python numpy mediapipe scikit-learn matplotlib pandas
```

## Usage

### Training the Model (Optional)

Place your training and testing datasets in appropriate folders (e.g., `dataset/train_set` and `dataset/test_set`), organized into subfolders named "man" and "woman."

Run the training script:

```bash
python train_gender_model.py
```

This will train the CNN model and save the model architecture (`GenderModel.json`) and weights (`GenderModel.weights.h5`) to the `models` directory.

### Running Real-Time Detection

Ensure the trained model files (`GenderModel.json` and `GenderModel.weights.h5`) are in the `models` directory.

Run the main detection script:

```bash
python gender_detection.py
```

The webcam feed will be displayed with bounding boxes around detected faces and their predicted genders. Press `q` to exit.

## Project Structure

```
gender-detection/
├── dataset/
│   ├── train_set/
│   │   ├── man/
│   │   └── woman/
│   └── test_set/
│       ├── man/
│       └── woman/
├── models/
│   ├── GenderModel.json
│   └── GenderModel.weights.h5
├── gender_detection.py
├── train_gender_model.py
├── README.md
└── requirements.txt
```

## Evaluation

The training script includes evaluation on the test set, producing metrics like accuracy and a confusion matrix.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgements

* MediaPipe for face detection.
* TensorFlow/Keras for deep learning.

## Further Improvements

* Improve model accuracy with a larger and more diverse dataset.
* Implement face tracking for smoother results.
* Explore different CNN architectures.
* Add confidence scores to predictions.

