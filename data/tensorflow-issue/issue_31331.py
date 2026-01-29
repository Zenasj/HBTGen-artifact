# tf.random.uniform((B, 128, 128, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers as per the example model with BatchNormalization
        self.conv = tf.keras.layers.Conv2D(4, (3, 3))
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, inputs, training=False):
        # Apply layers sequentially
        x = self.conv(inputs)
        # BatchNormalization layer respects the training flag
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile as in the example (optimizer, loss, metrics)
    # This is typically done outside for training, but we include for completeness
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def GetInput():
    # Return a random input tensor matching the input shape of the model
    # Batch size is assumed to be 1 for example
    # Input shape from the issue: (128, 128, 1)
    batch_size = 1
    input_shape = (batch_size, 128, 128, 1)
    # Use float32 dtype as typical for images
    return tf.random.uniform(input_shape, dtype=tf.float32)

