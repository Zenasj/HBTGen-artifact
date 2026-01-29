# tf.random.uniform((256, 144, 130), dtype=tf.float32) ← Input shape inferred from the issue's example batch_size=256, num_tsteps=144, num_features=130

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        batch_size = 256
        num_tsteps = 144
        num_features = 130
        num_units = 88
        
        # Masking layer to skip padded zeros (mask_value=0.0)
        self.masking = tf.keras.layers.Masking(mask_value=0.0,
                                               input_shape=(num_tsteps, num_features))
        # LSTM layer with fixed batch shape (used batch_input_shape in Keras Sequential example)
        # cuDNN LSTM enabled by default on GPU if conditions met
        self.lstm = tf.keras.layers.LSTM(num_units,
                                         return_sequences=True,
                                         stateful=False)
        # TimeDistributed Dense with sigmoid activation for binary outputs per timestep
        self.time_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
    
    def call(self, inputs, training=False):
        """
        Forward pass:
        1) Mask input padded zeros
        2) Process sequence with LSTM
        3) Predict per timestep via TimeDistributed dense + sigmoid activation
        
        Note: Based on the issue's discussion, cuDNN LSTM + Masking on GPU throws errors if
        entire sequences in batch are fully masked (zero). This model assumes the user handles 
        input batching to avoid fully zero samples.
        """
        x = self.masking(inputs)
        x = self.lstm(x)
        x = self.time_dense(x)
        x = self.sigmoid(x)
        return x

def my_model_function():
    """
    Instantiate and return the MyModel instance.
    No pretrained weights are loaded since example focuses on model structure and usage.
    """
    return MyModel()

def GetInput():
    """
    Return a random input tensor matching the expected input shape for MyModel.
    Batch size, time steps, and feature dimensions inferred from issue's example.
    Uses float32 dtype as typical for TensorFlow models.
    """
    batch_size = 256
    num_tsteps = 144
    num_features = 130
    return tf.random.uniform(shape=(batch_size, num_tsteps, num_features), dtype=tf.float32)

