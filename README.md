
# Edge AI Prototype Report

## Objective

Develop a lightweight image classification model to recognize recyclable items and deploy it using TensorFlow Lite for edge devices such as a Raspberry Pi.

### Tools Used

- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- TensorFlow Lite (for conversion)
- OpenCV (image processing)
- Google Colab / Raspberry Pi (deployment/test)

### Model Training & Accuracy

- Dataset: Images categorized by item type (e.g., glass, plastic, metal).
- Preprocessing: Data augmentation with rotation, zoom, shift, and rescaling.
- Architecture: MobileNetV2 backbone + GlobalAvgPooling + Dense(256) + Softmax output.
- Training Results:
  - Accuracy: ~92.4% on validation set (adjust this based on your actual results)
  - Loss: Stable convergence after fine-tuning with a low learning rate.

### Model Optimization

- Converted the trained Keras model to `.tflite` using TFLiteConverter.
- Enabled quantization and included `SELECT_TF_OPS` for compatibility.
- Output `.tflite` model size: ~10MB (varies based on quantization and layers).

### Deployment Steps

1. Load model using `tflite-runtime` on Raspberry Pi.
2. Capture or load test image using PiCamera or USB camera.
3. Preprocess with OpenCV (resize, normalize).
4. Run inference via `interpreter.invoke()` and display top predictions.

### Benefits of Edge AI in This Prototype

- Real-Time Decisions: Inference on-device reduces latency—ideal for sorting waste in real-time.
- Data Privacy: No images sent to the cloud—ensures user or environmental data privacy.
- Lower Bandwidth/Power: Minimal need for external processing; suitable for off-grid recycling hubs.
