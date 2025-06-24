# Edge AI Food Recognition

A lightweight, privacy-first image classification system for real-time food item recognition in smart kitchens. Built with TensorFlow 2.x and TensorFlow Lite, this project empowers dietary tracking and kitchen automation directly on edge devices—no cloud or internet required.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Inference](#inference)
- [TensorFlow Lite Conversion](#tensorflow-lite-conversion)
- [Edge Deployment](#edge-deployment)
- [Benefits of Edge AI](#benefits-of-edge-ai)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **MobileNetV2 backbone** for ultra-efficient inference on edge hardware
- **Data augmentation** pipeline for improved generalization
- **Top-1 / Top-3 accuracy metrics**
- **One-command TensorFlow Lite export** for edge deployment
- **Privacy-by-design:** all operations run locally, never in the cloud

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd edge-ai-food-recognition
pip install -r requirements.txt
```

---

## Quick Start

1. **Prepare your dataset**  
   Structure your data with each food class in its own subfolder under `train/` and `validation/`.

2. **Preprocess Data**  
   Resize, normalize, and augment your images:
   ```bash
   python src/data/preprocess.py
   ```

3. **Train the Model**  
   Start training:
   ```bash
   python src/training/train.py
   ```
   The best model will be saved to your configured path.

4. **Convert to TensorFlow Lite**  
   Export your trained model for edge deployment:
   ```bash
   python src/tflite/convert.py
   ```

5. **Run Inference**  
   Run predictions on images:
   ```bash
   python src/inference/infer.py
   ```

---

## Model Architecture

- **Backbone:** MobileNetV2 (ImageNet pre-trained, `include_top=False`)
- **Head:** GlobalAveragePooling2D → Dense(256, relu) → Dense(num_classes, softmax)
- **Transfer Learning:** Freeze base initially, then unfreeze for fine-tuning

Explore the architecture in [`src/models/model.py`](src/models/model.py).

---

## Data Preprocessing

- **Resize:** 224x224 pixels
- **Normalize:** Scale pixel values to [0, 1]
- **Augment:** Random horizontal flip, rotation, zoom

See preprocessing script: [`src/data/preprocess.py`](src/data/preprocess.py)

---

## Training

- **Script:** [`src/training/train.py`](src/training/train.py)
- **Procedure:**  
  1. Train the classifier head (base frozen)  
  2. Fine-tune the entire model  
  3. Save the best model checkpoint

---

## Inference

- **Script:** [`src/inference/infer.py`](src/inference/infer.py)
- **Output:** Top-3 predicted classes with softmax confidence scores

---

## TensorFlow Lite Conversion

- **Script:** [`src/tflite/convert.py`](src/tflite/convert.py)
- **Output:** Compact `.tflite` model (~6MB) for edge devices

---

## Edge Deployment

- Deploy your `.tflite` model on a Raspberry Pi or similar hardware.
- Use the TensorFlow Lite runtime for on-device inference.
- Example inference speed: ~110 ms/image (Raspberry Pi simulation)

---

## Benefits of Edge AI

| Benefit    | Description                                                      |
|------------|------------------------------------------------------------------|
| **Privacy**   | All data stays local; no cloud upload or external processing   |
| **Speed**     | Real-time feedback for grocery logging and meal prep           |
| **Efficiency**| Optimized for low-resource systems (e.g., Raspberry Pi, IoT)  |

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to help improve this project.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
