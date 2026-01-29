# tf.random.uniform((B, L, 1), dtype=tf.float32)  # Assumed input shape for 1D conv model (batch, length, channels=1)

import tensorflow as tf
from tensorflow.keras import layers, Model, initializers, regularizers

class MyModel(tf.keras.Model):
    def __init__(self, input_length, model_output_len, **kwargs):
        super().__init__(**kwargs)
        # Based on provided conv1d model creating a keras.Model equivalent in subclassing style
        
        # Initializers as per original code
        kernel_init = initializers.GlorotUniform(seed=33)
        bias_init = initializers.HeUniform(seed=33)
        l2_reg = regularizers.L2(0.01)
        
        # Conv1D layers
        self.conv1 = layers.Conv1D(
            filters=128, kernel_size=8, strides=2, padding='same',
            activation='elu', kernel_initializer=kernel_init,
            bias_initializer=bias_init, name='Conv1D_1')
        
        self.conv2 = layers.Conv1D(
            filters=64, kernel_size=4, strides=1, padding='same',
            activation='elu', kernel_initializer=kernel_init,
            bias_initializer=bias_init, name='Conv1D_2')
        
        # Flatten layer
        self.flatten = layers.Flatten()
        
        # Dense layers with ELU, dropout
        self.dense1 = layers.Dense(
            units=512, kernel_initializer=kernel_init,
            bias_initializer=bias_init, kernel_regularizer=l2_reg,
            name='Dense_1')
        self.elu1 = layers.ELU()
        self.dropout = layers.Dropout(0.33, seed=33)
        self.dense2 = layers.Dense(
            units=512, activation='elu', kernel_initializer=kernel_init,
            bias_initializer=bias_init, kernel_regularizer=l2_reg,
            name='Dense_2')
        
        # Output layer with sigmoid
        self.output_layer = layers.Dense(
            units=model_output_len, activation='sigmoid', name='output')
        
        # Store input length for signature / shape validation (optional)
        self.input_length = input_length
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.elu1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        out = self.output_layer(x)
        return out

def my_model_function():
    # Assumptions since original X_train_pooled_output shape unknown,
    # Let's assume input length 128 (arbitrary) and output length 10.
    # These would typically be defined by your dataset.
    input_length = 128    # example input sequence length (time steps)
    model_output_len = 10 # example output dimension
    
    model = MyModel(input_length=input_length, model_output_len=model_output_len)
    
    # Build model by calling it once with expected input shape (batch, length, 1)
    dummy_input = tf.zeros([1, input_length, 1], dtype=tf.float32)
    model(dummy_input, training=False)
    
    # Optionally compile for usage
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True, num_thresholds=5000)]
    )
    return model

def GetInput():
    # Generate a batch of inputs with shape matching model input (B, input_length, 1).
    # Using batch size 8 as in the example.
    batch_size = 8
    input_length = 128  # must match MyModel input_length
    # Generate random float inputs
    return tf.random.uniform(shape=(batch_size, input_length, 1), dtype=tf.float32)

