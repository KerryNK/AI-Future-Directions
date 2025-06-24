import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

# --- Parameters ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'path/to/train/data'           # <-- Update with your training dataset path
VALIDATION_DIR = 'path/to/validation/data' # <-- Update with your validation dataset path
MODEL_SAVE_PATH = 'food_classifier_model.h5'
TFLITE_MODEL_PATH = 'food_classifier_model.tflite'
CLASS_NAMES_PATH = None  # Optional: path to class names txt file

# --- Data Generators with Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- Model Creation ---
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False  # Freeze base model

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate and train model
num_classes = len(train_generator.class_indices)
model = create_model(num_classes)

print("Starting training...")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# Optional fine-tuning of base model
print("Fine-tuning base model...")
base_model = model.layers[1]  # MobileNetV2 base model
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=5
)

# Save the trained Keras model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# --- Convert to TensorFlow Lite ---
def convert_to_tflite(keras_model_path, tflite_model_path):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

convert_to_tflite(MODEL_SAVE_PATH, TFLITE_MODEL_PATH)

# --- Inference Class ---
class FoodClassifier:
    def __init__(self, tflite_model_path, class_names_path=None):
        # Load TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        # Load class names
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # Use generic class names if none provided
            self.class_names = [f"class_{i}" for i in range(self.interpreter.get_output_details()[0]['shape'][1])]

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        top_k_indices = output_data.argsort()[-3:][::-1]
        top_k_scores = output_data[top_k_indices]
        return [(self.class_names[i], float(top_k_scores[idx])) for idx, i in enumerate(top_k_indices)]

# --- Example Usage ---
if __name__ == "__main__":
    # Update these paths before running inference
    test_image_path = 'path/to/test/image.jpg'
    tflite_model_path = TFLITE_MODEL_PATH
    class_names_path = CLASS_NAMES_PATH

    classifier = FoodClassifier(tflite_model_path, class_names_path)
    predictions = classifier.predict(test_image_path)

    print("Top 3 Predictions:")
    for class_name, score in predictions:
        print(f"{class_name}: {score:.4f}")
