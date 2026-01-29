# tf.random.uniform((128, 28, 28), dtype=tf.float32) ← Assumed input shape and dtype based on MNIST batch size and preprocessing
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Build a model matching the described MNIST CNN
        self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Instantiate the model. No pretrained weights available from the issue; just create fresh instance.
    return MyModel()

def GetInput():
    # Return a random input tensor matching (batch_size, height, width) as expected by the model
    # Batch size 128 inferred from ds_train batch size.
    return tf.random.uniform((128, 28, 28), dtype=tf.float32)

