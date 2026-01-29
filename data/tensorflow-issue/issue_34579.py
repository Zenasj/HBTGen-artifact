# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset (28x28 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture from issue reproducible code:
        # Sequential:
        # Flatten input_shape=(28, 28)
        # Dense(128) relu
        # Dropout(0.2)
        # Dense(10) softmax
        
        # Defining layers explicitly as submodules to fit tf.keras.Model subclass style
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile with same loss/optimizer/metrics as original in issue
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Train weights are not loaded here since no data given,
    # but user can train or load weights externally.
    return model

def GetInput():
    # Return a batch of dummy MNIST-like inputs matching model input shape
    # Batch size is arbitrarily chosen as 32
    # Input tensor elements normalized [0, 1] float32, shape (B, 28, 28)
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

