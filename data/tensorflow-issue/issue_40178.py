# tf.random.uniform((B, 56, 40), dtype=tf.float32)  ← input shape inferred from the issue's x_train shape (batch, 56, 40)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the reported model architecture in the issue:
        # Input: (56, 40)
        # Reshape -> (56, 40, 1)
        # Conv2D(8 filters, (20, 5) kernel, strides=(1,2)) -> output shape (37, 18, 8)
        # Reshape -> (37, 144)
        # GRU(20 units)
        # Dense(2 units)
        self.reshape1 = tf.keras.layers.Reshape((56, 40, 1))
        self.conv2d = tf.keras.layers.Conv2D(8, (20, 5), strides=(1, 2))
        self.reshape2 = tf.keras.layers.Reshape((37, 18 * 8))  # 18*8=144
        self.gru = tf.keras.layers.GRU(20)
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=None):
        x = self.reshape1(inputs)
        x = self.conv2d(x)
        x = self.reshape2(x)
        x = self.gru(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel properly initialized.
    model = MyModel()
    # Compile the model as in the example (the original used default compile with no args)
    # Because no loss/optimizer specified in the issue, compiling minimally without loss/optimizer
    # but to allow fit, provide loss and optimizer:
    model.compile(optimizer='adam', loss='mse')
    return model

def GetInput():
    # Generate random float32 input tensor matching expected input: shape (batch, 56, 40)
    # Batch size can be chosen as 4 for example
    batch_size = 4
    return tf.random.uniform((batch_size, 56, 40), dtype=tf.float32)

