import tensorflow as tf
import numpy as np
import cv2
import os

class FoodClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = self.load_class_names()

    def load_class_names(self):
        # Load class names from a text file or define them directly
        return ["class1", "class2", "class3"]  # Replace with actual class names

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        predictions = self.model.predict(image)
        top_k = tf.nn.top_k(predictions, k=3)
        return top_k.indices.numpy()[0], top_k.values.numpy()[0]

def main(image_path, model_path):
    classifier = FoodClassifier(model_path)
    indices, scores = classifier.predict(image_path)
    
    print("Top 3 Predictions:")
    for i in range(len(indices)):
        print(f"{classifier.class_names[indices[i]]}: {scores[i]:.4f}")

if __name__ == "__main__":
    # Example usage
    model_path = 'path/to/your/model.h5'  # Update with the actual model path
    image_path = 'path/to/your/image.jpg'  # Update with the actual image path
    main(image_path, model_path)