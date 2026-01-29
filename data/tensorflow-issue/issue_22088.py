# tf.random.uniform((batch_size=100, height=88, width=200, channels=3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    """
    Simple CNN model from the provided issue with 3 Conv2D layers followed by a Dense layer
    and a single scalar prediction output layer.
    
    Assumptions:
    - Input shape inferred from input_fn: (100, 88, 200, 3) batch size 100, 
      height 88, width 200, channels 3.
    - Activations use tf.nn.relu6 as in original code.
    - Kernel initializer following original with uniform variance scaling.
    """
    def __init__(self, name=''):
        super(MyModel, self).__init__(name=name)
        
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / 3.0, distribution='uniform', mode='fan_in')

        self.conv1 = layers.Conv2D(
            32, (3, 3), activation=tf.nn.relu6, kernel_initializer=kernel_initializer, name='conv1')
        self.conv2 = layers.Conv2D(
            64, (3, 3), activation=tf.nn.relu6, kernel_initializer=kernel_initializer, name='conv2')
        self.conv3 = layers.Conv2D(
            64, (3, 3), activation=tf.nn.relu6, kernel_initializer=kernel_initializer, name='conv3')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(
            512, activation=tf.nn.relu6, name='fc1')
        self.steer_predictor = layers.Dense(1, name='steer_predictor')

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        output = self.steer_predictor(x)
        return output

def my_model_function():
    """
    Returns an instance of MyModel.

    If distributed training is needed, one can wrap this in distribution strategy later,
    but here we just return the model instance.
    """
    return MyModel()

def GetInput():
    """
    Returns a random tensor input matching the model input shape expected:
    shape = (100, 88, 200, 3), dtype float32, uniform distribution in [0, 1).

    This matches the original example input_fn features tensor for compatibility.
    """
    return tf.random.uniform(shape=(100, 88, 200, 3), dtype=tf.float32)

