# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST example (batch, height, width, channels)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same CNN architecture as in the MirroredStrategy MNIST example
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output, no activation here

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)  # logits for classification
        return x

def my_model_function():
    # Instantiate and compile the model as done within the MirroredStrategy scope.
    model = MyModel()
    # Compile model with SparseCategoricalCrossentropy from logits and Adam optimizer to resemble the original example
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # Note: weights not loaded here; user can train or load weights externally.
    return model

def GetInput():
    # Return a batch of random input images matching MNIST shape with batch size 256 (64 per replica * 4 replicas)
    # Using dtype float32 values in [0,1]
    BATCH_SIZE = 64 * 4
    return tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)

