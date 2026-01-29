# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← input shape inferred from mnist dataset (grayscale 28x28 images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten input from (28,28) to (784,)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer 128 units + ReLU
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        # Note: The original bug report mentioned Dropout causing issues in multi-worker or ray context,
        # so it's commented out here to avoid the reported reduction error.
        # self.dropout = tf.keras.layers.Dropout(0.2)
        # Output layer with 10 logits (for 10 classes)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        # Dropout commented out due to known issues from the report
        # if training:
        #     x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()

    # Compile the model with a suitable optimizer and loss, as per the repro
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    return model

def GetInput():
    # Return a random float32 tensor of shape (batch_size, 28, 28)
    # Batch size chosen as 32 (arbitrary typical batch size)
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

