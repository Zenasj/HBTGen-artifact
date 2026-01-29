# tf.random.uniform((64*2, 28, 28, 1), dtype=tf.float32) ← inferred input shape: batch size=GLOBAL_BATCH_SIZE=64*2, 28x28 grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CNN model as per build_and_compile_cnn_model from the issue
        # Using Sequential layers inside functional style here for clarity and flexibility
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1))
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.conv2d(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Build and compile the CNN model, matching loss, optimizer and metrics from the issue
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor input matching the expected input shape for MyModel
    # The input shape is batch size (GLOBAL_BATCH_SIZE=64*2), 28x28 images with 1 channel (grayscale)
    batch_size = 64 * 2  # inferred from NUM_WORKERS=2 and BATCH_SIZE=64
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

