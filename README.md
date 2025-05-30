# Fruit Freshness Detector

A desktop application that uses deep learning to detect whether fruits and vegetables are fresh or rotten. Built with TensorFlow and CustomTkinter, this application provides a user-friendly interface for food freshness detection along with nutritional information.

## Features

- Detects freshness of various fruits and vegetables
- Modern dark-themed UI
- Real-time freshness prediction
- Displays nutritional information for detected food items
- Supports multiple food types including:
  - Fruits: Apple, Banana, Mango, Orange, Strawberry
  - Vegetables: Bell Pepper, Bitter Gourd, Capsicum, Carrot, Cucumber, Okra, Potato, Tomato

## Requirements

- Python 3.7+
- TensorFlow 2.x
- CustomTkinter
- PIL (Python Imaging Library)
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/fruit-freshness-detector.git
cd fruit-freshness-detector
```

2. Install the required packages:
```bash
pip install tensorflow customtkinter pillow numpy matplotlib
```

3. Download the model file:
Due to size limitations, the model file (`fruit_freshness_model_best.h5`) is not included in the repository. Please download it from [add your model file link] and place it in the project root directory.

## Usage

1. Run the application:
```bash
python fresh_rotten_detector.py
```

2. Click the "Upload Image" button to select an image of a fruit or vegetable
3. The application will display the freshness prediction and nutritional information

## Model Information

The application uses MobileNetV2 architecture for food type classification and freshness detection. The model has been trained on a dataset of fresh and rotten fruits and vegetables.
