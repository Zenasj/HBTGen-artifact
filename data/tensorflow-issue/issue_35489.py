# tf.random.uniform((B, 28*28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple dense neural network consistent with the issue's example
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', input_shape=(784,))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel, compiled in the same way as the example
    model = MyModel()
    # Compile must be called explicitly outside the model class in TF 2.x if using model.fit
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a batch of random float32 tensor with shape (B, 784) matching flattened 28x28 greyscale images
    # Batch size arbitrarily chosen as 32, can be any positive integer
    batch_size = 32
    # Inputs normalized similar to MNIST scaling to [0,1], so uniform 0-1 floats makes sense here
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)

