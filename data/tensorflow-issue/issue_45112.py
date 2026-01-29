# tf.random.uniform((B, None, 5), dtype=tf.float32) ← Batch size B varies, sequence length None (variable), features 5

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Inputs accept variable-length sequences with 5 features.
        # Ragged tensors are unsupported by Conv1DTranspose, so inputs are dense with padding.
        self.input_1_layer = keras.Input(shape=(None, 5), name="input_1")
        self.input_2_layer = keras.Input(shape=(None, 5), name="input_2")
        
        # Conv1DTranspose layers require dense tensor inputs with uniform length.
        # We create two submodels that process each input independently.
        self.conv1d_transpose_1 = layers.Conv1DTranspose(
            filters=16, kernel_size=3, padding='same', activation='relu', name="conv1dt_1"
        )
        self.conv1d_transpose_2 = layers.Conv1DTranspose(
            filters=16, kernel_size=3, padding='same', activation='relu', name="conv1dt_2"
        )

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # inputs are expected to be a tuple of two tensors:
        # (input_1: [batch, seq_len, 5], input_2: [batch, seq_len, 5])
        input_1, input_2 = inputs
        
        # Forward pass through each Conv1DTranspose layer
        output_1 = self.conv1d_transpose_1(input_1)
        output_2 = self.conv1d_transpose_2(input_2)
        
        # Return tuple of outputs
        return (output_1, output_2)

def my_model_function():
    # Instantiate and compile the model with a custom loss. We compile here to align with original example.
    model = MyModel()
    
    # Construct model inputs so model can be built and weights initialized before return.
    # This is necessary because raw tf.keras.Model.compile expects built model.
    dummy_input_1 = tf.random.uniform((1, 4, 5), dtype=tf.float32)
    dummy_input_2 = tf.random.uniform((1, 4, 5), dtype=tf.float32)
    model((dummy_input_1, dummy_input_2))  # Build model once with sample input
    
    # Define custom loss as mean absolute error (like in original example)
    def custom_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_pred - y_true))
    
    # Compile model with SGD and custom losses for each output
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss=[custom_loss, custom_loss]
    )
    
    return model

def GetInput():
    # Return a tuple of two random tensors with shape [batch, sequence length, features=5]
    # Sequence length is variable; here we pick 3 and 5 to simulate variable lengths padding
    # Since Conv1DTranspose requires dense tensor input, variable length sequences must
    # be padded to uniform lengths in each batch.
    
    # For inference demonstration, let's pick batch size = 2 and uniform padded length = 6
    batch_size = 2
    max_seq_len = 6
    feature_dim = 5
    
    # Random uniform tensor simulating padded input (e.g. zeros padding)
    input_1 = tf.random.uniform((batch_size, max_seq_len, feature_dim), dtype=tf.float32)
    input_2 = tf.random.uniform((batch_size, max_seq_len, feature_dim), dtype=tf.float32)
    
    # Normally you'd also supply a mask or handle padding downstream.
    return (input_1, input_2)

