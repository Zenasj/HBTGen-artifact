# tf.random.uniform((B, 28*28), dtype=tf.float32) ← Assumed input shape from MNIST example flattened image

import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reshape the flat input to 28x28 image shape
        self.reshape = l.Reshape(target_shape=(28, 28))
        # Dense layer with softmax activation applied to each row (28 timesteps) with output dim = 10 classes
        self.dense = l.Dense(10, activation='softmax')
        # Lambda layer wrapping the custom CTC loss function
        self.ctc_lambda = l.Lambda(self.ctc_batch_cost, output_shape=(1,), name='ctc')

    def ctc_batch_cost(self, args):
        """
        Custom CTC batch cost function compatible with the signature expected for Lambda layer.
        
        Args:
            args: tuple of (y_pred, y_true)
              - y_pred: predicted softmax outputs, shape (batch, time, num_classes)
              - y_true: ground truth labels and lengths concatenated in input tensor of shape (batch, input_dim)
              
        Returns:
            Tensor of shape (batch, 1) representing the CTC loss for each sample.
        """
        y_pred, y_true = args

        # NOTE: The original issue code takes input_length and label_length from y_true indices + 5,
        # which is a hacky placeholder that likely was for demonstration. Here, we follow the same logic.

        # Extract input_length and label_length from y_true assuming format:
        # y_true shape = (batch, 28*28), flatten MNIST image input used both as input and dummy "labels"
        # For demonstration, use y_true[:, 1:2] + 5 as lengths with shape (batch,1)
        # In a real scenario, you would pass actual label sequences with their lengths.
        input_length = y_true[:, 1:2] + 5
        label_length = y_true[:, 1:2] + 5

        # Call Keras CTC batch cost (TensorFlow backend)
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def call(self, inputs, training=None):
        # Inputs shape: (batch, 28*28), float32 tensor
        x = self.reshape(inputs)                   # (batch, 28, 28)
        x = self.dense(x)                          # (batch, 28, 10) - softmax over last dim
        # output of Lambda is shape (batch, 1) with CTC loss per sample
        loss = self.ctc_lambda([x, inputs])
        return loss

def my_model_function():
    """
    Creates an instance of MyModel.
    Note:
      - This model is built similar to the MNIST TPU example with a custom CTC loss Lambda layer,
        reflecting the issue's original code.
      - Inputs expected to be batches of flattened images (28*28).
    """
    return MyModel()

def GetInput():
    """
    Returns a random tensor input matching MyModel expected input:
      Shape: (batch_size, 28*28)
      Dtype: float32
    
    This simulates flattened MNIST images as in the original issue example.
    """
    batch_size = 2  # Arbitrary small batch size for testing
    # Random uniform in [0, 1)
    return tf.random.uniform((batch_size, 28*28), dtype=tf.float32)

