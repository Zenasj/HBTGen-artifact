# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Inferred typical input: batch of images with shape (batch, height, width, channels)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MNIST-like classification model per user example snippet
        # This model mirrors the Sequential model from the issue's tested code with CC 3.0 enabled
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)  # 10 classes

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of the model suitable for compilation/training inference
    model = MyModel()
    # Compile with the loss and optimizer setup used in the example
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def GetInput():
    # Generates random input tensor matching the MNIST grayscale image with shape (batch=1, 28, 28)
    # dtype float32 and normalized in [0, 1]
    return tf.random.uniform((1, 28, 28), minval=0, maxval=1, dtype=tf.float32)

