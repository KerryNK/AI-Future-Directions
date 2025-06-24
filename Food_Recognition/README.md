# Edge AI Food Recognition

A lightweight, privacy-focused image classification solution for food item recognition in smart kitchens. Built with TensorFlow 2.x and TensorFlow Lite, this project enables real-time dietary tracking and kitchen automation on edge devices—no cloud required.

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

- **Lightweight MobileNetV2 backbone** for efficient edge inference
- **Data augmentation** for robust generalization
- **Top-1 and Top-3 accuracy reporting**
- **TensorFlow Lite conversion** for edge deployment
- **Privacy-first:** all processing is local

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd edge-ai-food-recognition
pip install -r requirements.txt
```

---

## Quick Start

1. **Prepare your dataset**  
   Organize images in subfolders by class under `train/` and `validation/`.

2. **Preprocess Data**  
   Use `src/data/preprocess.py` to resize, normalize, and augment images.

3. **Train the Model**  
   Run:
   ```bash
   python src/training/train.py
   ```
   The trained model will be saved to your specified path.

4. **Convert to TensorFlow Lite**  
   ```bash
   python src/tflite/convert.py
   ```

5. **Run Inference**  
   ```bash
   python src/inference/infer.py
   ```

---

## Model Architecture

- **Backbone:** MobileNetV2 (pretrained on ImageNet, `include_top=False`)
- **Head:** GlobalAveragePooling2D → Dense(256, relu) → Dense(num_classes, softmax)
- **Transfer Learning:** Base layers frozen initially, then fine-tuned

See [`src/models/model.py`](src/models/model.py) for details.

---

## Data Preprocessing

- **Resize:** 224x224 pixels
- **Normalize:** Pixel values scaled to [0, 1]
- **Augmentation:** Horizontal flip, rotation, zoom

See [`src/data/preprocess.py`](src/data/preprocess.py).

---

## Training

- **Script:** [`src/training/train.py`](src/training/train.py)
- **Process:**  
  1. Train classifier head with base frozen  
  2. Fine-tune base layers  
  3. Save best model

---

## Inference

- **Script:** [`src/inference/infer.py`](src/inference/infer.py)
- **Output:** Top-3 class predictions with softmax confidence scores

---

## TensorFlow Lite Conversion

- **Script:** [`src/tflite/convert.py`](src/tflite/convert.py)
- **Result:** `.tflite` model (~6MB), ready for edge deployment

---

## Edge Deployment

- Deploy the `.tflite` model to a Raspberry Pi or similar device.
- Use TensorFlow Lite runtime for inference.
- Example inference time: ~110 ms/image (Raspberry Pi simulation)

---

## Benefits of Edge AI

| Benefit   | Description                                                         |
|-----------|---------------------------------------------------------------------|
| Privacy   | All data stays local—no cloud upload required                       |
| Speed     | Instant feedback for grocery logging and meal prep                  |
| Efficiency| Optimized for low-resource devices (e.g., Raspberry Pi, smart fridge)|

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

MIT License. See [LICENSE](LICENSE) for details.