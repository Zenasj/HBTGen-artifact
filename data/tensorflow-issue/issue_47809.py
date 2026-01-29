# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST example (batch, height, width)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer converts (28,28) input to (784,)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Hidden dense layer with 128 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Dropout layer with rate 0.2 
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Output dense layer with 10 units (for 10 classes)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Create and compile model with Adam optimizer, SparseCategoricalCrossentropy loss, and accuracy metric
    model = MyModel()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def GetInput():
    # Return a batch of 1 example, 28x28 with float32 values between 0 and 1
    # Matches input expected by MyModel
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

