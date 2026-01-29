# tf.random.uniform((1, 10, 32, 32, 1), dtype=tf.float32) ← Inferred input shape: batch=1, time=10, height=32, width=32, channels=1
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        kernel_size = 3
        nfilters_lstm = 16
        
        # ConvLSTM2D layer with channels_last, returning last output (no sequence)
        self.convlstm2d = tf.keras.layers.ConvLSTM2D(
            filters=nfilters_lstm,
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            data_format='channels_last',
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            return_sequences=False,
            name='conv_lstm2d'
        )
        self.batchnorm = tf.keras.layers.BatchNormalization(name='batchnorm')
        self.conv2d = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            activation='relu',
            name='conv2d_out'
        )
        
    def call(self, inputs, training=False):
        """
        Args:
            inputs: Tensor with shape (batch, time, height, width, channels), e.g. (1,10,32,32,1)
        Returns:
            Tensor with shape (batch, height, width, 1), e.g. (1,32,32,1)
        """
        x = self.convlstm2d(inputs, training=training)
        x = self.batchnorm(x, training=training)
        x = self.conv2d(x)
        return x

def my_model_function():
    """
    Builds and returns an instance of MyModel.
    
    Notes:
    - The model here matches the structure from the issue: input with shape (1,10,32,32,1),
      ConvLSTM2D layer with 16 filters and 3x3 kernel,
      followed by BatchNormalization and a Conv2D producing 1 output channel with ReLU.
    - This model is compatible with TF 2.20.0, and can be compiled with XLA jit_compile.
    - For TFLite conversion involving ConvLSTM2D, note that TFLite may require using SELECT_TF_OPS.
    """
    model = MyModel()
    # Build the model by calling on sample input to infer shapes
    dummy_input = tf.random.uniform((1, 10, 32, 32, 1), dtype=tf.float32)
    _ = model(dummy_input)
    return model

def GetInput():
    """
    Returns a sample valid random input tensor for MyModel.
    Shape: (batch=1, time=10, height=32, width=32, channels=1)
    dtype: float32
    """
    return tf.random.uniform(shape=(1, 10, 32, 32, 1), dtype=tf.float32)

