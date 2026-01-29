# tf.random.uniform((1, 25, 40, 1), dtype=tf.float32) ← inferred input shape from the issue comments

import tensorflow as tf
import time

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original code loads a model from disk.
        # Since that is not provided, we reconstruct a minimal example model
        # with similar input shape and a binary output (sigmoid) modeling
        # a small CNN for binary classification.

        # Assumption: A small conv net mimicking the referenced resnet32's input shape and output type.
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.out_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out_layer(x)

    def predict_on_batch(self, x):
        # Adds compatibility for original predict_on_batch usage
        return self(x, training=False)

def my_model_function():
    # Return an instance of MyModel
    # Normally weights would be loaded if available; here created fresh
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape: (1, 25, 40, 1)
    # dtype float32 as typical for images / feature maps
    return tf.random.uniform((1, 25, 40, 1), dtype=tf.float32)

# Additional helper class derived from the issue for prediction timing and thresholding
class PredictPerFeat:
    def __init__(self, model):
        # model: tf.keras.Model instance
        self.model = model

    def predict(self, feats):
        # Measure time taken for prediction on a single batch tensor feats
        # feats shape expected: (1, 25, 40, 1)
        start_time = time.time()
        out = self.model.predict_on_batch(feats)
        elapsed = time.time() - start_time
        print("--- %s seconds ---" % elapsed)
        # Output thresholding to integer label as in original code: int(out > 0.5)
        out = int(tf.reduce_mean(out) > 0.5)  # reduce_mean to handle possible batch dim
        return out

