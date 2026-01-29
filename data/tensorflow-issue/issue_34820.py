# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST data (28x28 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer: converts 28x28 images to 784 vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer with 128 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Dropout layer with 0.2 rate for regularization during training
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Output dense layer with 10 units and softmax for class probabilities
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        # Pass `training` flag to dropout to enable/disable dropout during inference
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel.
    model = MyModel()
    # Compile the model as done in the original example to match usage pattern
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a batch of random inputs with shape [batch_size, 28, 28]
    # Use batch size 32 as a common default
    batch_size = 32
    # Generate random floats between 0 and 1, dtype float32 consistent with normalized image data
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

