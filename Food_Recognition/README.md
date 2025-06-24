# Edge AI Food Recognition

This project is a lightweight image classification model designed for food item recognition in smart kitchens, utilizing TensorFlow 2.x and TensorFlow Lite. The focus is on privacy and real-time applications, making it suitable for edge deployment.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Inference](#inference)
- [TensorFlow Lite Conversion](#tensorflow-lite-conversion)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd edge-ai-food-recognition
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Use the `src/data/preprocess.py` script to load and preprocess your image dataset. This includes resizing images, normalizing pixel values, and applying data augmentation techniques.

2. **Training the Model**: Run the training script located at `src/training/train.py` to train the model on your dataset. The trained weights will be saved for later use.

3. **Running Inference**: After training, use the `src/inference/infer.py` script to make predictions on new images. Ensure the images are preprocessed as required.

4. **TensorFlow Lite Conversion**: Convert the trained model to TensorFlow Lite format using the `src/tflite/convert.py` script for deployment on edge devices.

## Model Architecture

The model is defined in `src/models/model.py`. It utilizes MobileNetV2 architecture, which is efficient for mobile and edge applications. The model includes methods for loading the pretrained model, adding GlobalAveragePooling and Dense layers, and compiling the model.

## Data Preprocessing

The preprocessing functions in `src/data/preprocess.py` handle loading and preparing the image data for training, including resizing to 224x224 pixels, normalizing pixel values to the range [0, 1], and applying augmentation techniques such as horizontal flip, rotation, and zoom.

## Training

The training process is managed in `src/training/train.py`, where the model is trained, evaluated, and the weights are saved. The training includes an initial phase with frozen base layers followed by fine-tuning.

## Inference

The inference script `src/inference/infer.py` allows users to load images, preprocess them, and obtain predictions from the trained model, outputting the top-3 class predictions along with softmax confidence scores.

## TensorFlow Lite Conversion

The conversion script `src/tflite/convert.py` facilitates the transformation of the trained Keras model into TensorFlow Lite format, enabling efficient deployment on edge devices.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.