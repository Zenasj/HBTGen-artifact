# tf.random.uniform((B, input_dim), dtype=tf.float32) ← assuming input is batch of vectors with shape (batch_size, input_dim)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Components of the original "State Model"
        self.state_reshape = tf.keras.layers.Reshape((input_dim, 1))
        self.state_lstm = tf.keras.layers.LSTM(64, return_sequences=False)
        self.state_dropout1 = tf.keras.layers.Dropout(0.3)
        self.state_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.state_dropout2 = tf.keras.layers.Dropout(0.3)
        self.state_dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

        # Components of the original "Temperature Model"
        self.temp_reshape = tf.keras.layers.Reshape((input_dim, 1))
        self.temp_lstm = tf.keras.layers.LSTM(64, return_sequences=False)
        self.temp_dropout1 = tf.keras.layers.Dropout(0.3)
        self.temp_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.temp_dropout2 = tf.keras.layers.Dropout(0.3)
        self.temp_dense2 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs, training=False):
        # inputs assumed shape (batch_size, input_dim)

        # Forward pass through State Model
        x_state = self.state_reshape(inputs)
        x_state = self.state_lstm(x_state)
        x_state = self.state_dropout1(x_state, training=training)
        x_state = self.state_dense1(x_state)
        x_state = self.state_dropout2(x_state, training=training)
        state_out = self.state_dense2(x_state)  # shape: (batch_size, 1), sigmoid output

        # Forward pass through Temperature Model
        x_temp = self.temp_reshape(inputs)
        x_temp = self.temp_lstm(x_temp)
        x_temp = self.temp_dropout1(x_temp, training=training)
        x_temp = self.temp_dense1(x_temp)
        x_temp = self.temp_dropout2(x_temp, training=training)
        temp_out = self.temp_dense2(x_temp)  # shape: (batch_size, output_dim), softmax output

        # Combine outputs into a dict for clarity (could also return tuple)
        # The user may want to compare or consume these separately
        # As per instruction, output should reflect the combined model,
        # so we return a dict with both outputs.
        return {'state_output': state_out, 'temp_output': temp_out}

def my_model_function():
    # Assumptions: input_dim and output_dim need to be set.
    # From original code:
    # - For State Model: input_dim is a parameter
    # - For Temp Model: input_dim and output_dim are parameters
    # Let's assume input_dim=10, output_dim=5 as a reasonable default
    input_dim = 10
    output_dim = 5
    model = MyModel(input_dim=input_dim, output_dim=output_dim)
    # The original code compiles models for training:
    # We mimic that here so model is ready to be trained or used
    model.compile(
        optimizer='adam',
        loss={'state_output': 'binary_crossentropy', 'temp_output': 'categorical_crossentropy'},
        metrics={'state_output': 'accuracy', 'temp_output': 'accuracy'}
    )
    return model

def GetInput():
    # Return a random input tensor compatible with the model's expected input shape:
    # Shape = (batch_size, input_dim)
    # Based on the assumption in my_model_function, input_dim=10
    batch_size = 4  # arbitrary batch size
    input_dim = 10
    # Random float input
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

