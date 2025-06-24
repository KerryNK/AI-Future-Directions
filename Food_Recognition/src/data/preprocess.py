import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    return image

def augment_data(images):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    augmented_images = []
    for image in images:
        image = np.expand_dims(image, axis=0)
        for batch in datagen.flow(image, batch_size=1):
            augmented_images.append(batch[0])
            if len(augmented_images) >= 5:  # Generate 5 augmented images per original image
                break
    return np.array(augmented_images)

def preprocess_dataset(image_paths):
    processed_images = []
    for image_path in image_paths:
        image = load_and_preprocess_image(image_path)
        processed_images.append(image)
    return np.array(processed_images)

# ...existing code...

def preprocess_dataset(image_paths):
    processed_images = []
    for image_path in image_paths:
        image = load_and_preprocess_image(image_path)
        processed_images.append(image)
    return np.array(processed_images)