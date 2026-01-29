# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape is 28x28 grayscale images with batch dimension B

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer to convert 2D input (28x28) into 1D vector (784,)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        
        # Hidden layer 1: Dense 300, ELU, He-normal init, L2 regularization 0.01
        self.hidden1 = tf.keras.layers.Dense(
            300,
            activation="elu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )
        
        # Hidden layer 2: Dense 100, ELU, He-normal init, L2 regularization 0.01
        self.hidden2 = tf.keras.layers.Dense(
            100,
            activation="elu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )
        
        # Output layer: Dense 10, softmax, L2 regularization 0.01
        self.output_layer = tf.keras.layers.Dense(
            10,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.output_layer(x)

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile the model as per original code
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="nadam",
        metrics=["accuracy"]
    )
    return model

def GetInput():
    # Generate a batch of random "images" with shape (batch_size=32, 28, 28) and float32 dtype
    # Values scaled like normalized pixel values roughly in [0,1] range
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

