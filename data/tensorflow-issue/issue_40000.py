from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
import random
import tensorflow_datasets as tfds

np.random.seed(42)
tf.random.set_seed(42)

train_ds, validation_ds = tfds.load(
    "tf_flowers",
    split=["train[:90%]", "train[90%:]"],
    as_supervised=True
)

size = (224, 224)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))

def normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
    shuffle(1024).\
    batch(32).\
    prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_ds = validation_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
    batch(32).\
    prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

base = MobileNet(weights="imagenet", include_top=False,
                    input_shape=(224, 224, 3))

def get_training_model():
    base.trainable = False
    class_head = base.output
    class_head = GlobalAveragePooling2D()(class_head)
    class_head = Dense(512, activation="relu")(class_head)
    class_head = Dropout(0.5)(class_head)
    class_head = Dense(5, activation="softmax")(class_head)

    classifier = Model(inputs=base.input, outputs=class_head)

    classifier.compile(loss="sparse_categorical_crossentropy", 
                          optimizer="adam",
                          metrics=["accuracy"])

    return classifier

test_model = get_training_model()
history = test_model.fit(train_ds,
              validation_data=validation_ds,
              epochs=5)

test_model_dir = "./test_model"
test_model.save(test_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(test_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()
f = open("test_model.tflite", "wb")
f.write(quantized_tflite_model)
f.close()

# Referred from: https://www.tensorflow.org/lite/performance/post_training_integer_quant
def evaluate_model(interpreter):
    accurate_count = 0

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    predictions = []
    for (val_images, val_labels) in validation_ds:
        for val_image, val_label in zip(val_images, val_labels):
            val_image = tf.expand_dims(val_image, 0)
            interpreter.set_tensor(input_index, val_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            probability = interpreter.get_tensor(output_index)
            flower_id = np.argmax(probability[0])
            predictions.append(flower_id)

            # Compare prediction results with ground truth labels to calculate accuracy.
            if flower_id == val_label:
                accurate_count += 1
    
    accuracy = accurate_count * 1.0 / len(predictions)

    return accuracy

interpreter_test = tf.lite.Interpreter(model_path="test_model.tflite")
interpreter_test.allocate_tensors()

accuracy = evaluate_model(interpreter_test)
print("accuracy is {}".format(accuracy))

accuracy is 0.4332425068119891