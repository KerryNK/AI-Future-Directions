import tensorflow as tf

def convert_model(model_path, tflite_model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Create a TFLiteConverter object from the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set optimization for size and performance
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model to TensorFlow Lite format
    tflite_model = converter.convert()

    # Save the converted model to a file
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    # Example usage
    convert_model('path/to/your/model.h5', 'path/to/save/model.tflite')