from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.lite as tflite
import os

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
DIRECTORY_PATH = "./archive/Dataset"  # Update this to the path of your dataset
LABELS_FILE = "labels.txt"

# Load the data
train_dataset = image_dataset_from_directory(
    DIRECTORY_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    DIRECTORY_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Prefetching for better performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ]
)

# Manually extract class names from the directory structure
class_names = sorted(item.name for item in os.scandir(DIRECTORY_PATH) if item.is_dir())

# Save the labels to a file
with open(LABELS_FILE, "w") as f:
    for label in class_names:
        f.write(f"{label}\n")

print(f"Labels saved to {LABELS_FILE}")

# Build the model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# Convert the model to TensorFlow Lite format with quantization
converter = tflite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = "model_quantized.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Quantized model saved to {tflite_model_path}")

import os

os.environ["TF_USE_LEGACY_KERAS"]="1"