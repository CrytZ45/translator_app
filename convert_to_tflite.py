import tensorflow as tf

# Define file paths
models = {
    "translator_model.tflite": "translator_model.h5",  # Kamayo → English
    "translator_model_reverse.tflite": "translator_model_reverse.h5"  # English → Kamayo
}

for tflite_filename, h5_filename in models.items():
    print(f"Converting {h5_filename} to {tflite_filename}...")

    # Load the Keras model
    model = tf.keras.models.load_model(h5_filename)

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)

    print(f"Saved: {tflite_filename}")

print("✅ Conversion complete!")
