# tf.random.uniform((B, 1, 1), dtype=tf.float32) x2 inputs for LSTM branches, output scalar sigmoid

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the two inputs each are (None, 1, 1)
        # Two LSTM branches with 40 units each
        self.lstm1 = tf.keras.layers.LSTM(40, return_sequences=False, name="lstm_input_x")
        self.lstm2 = tf.keras.layers.LSTM(40, return_sequences=False, name="lstm_input_y")
        # Concatenate and a dense output with sigmoid activation
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # inputs is expected to be either a list/tuple or dict with keys 'input_x' and 'input_y'
        # Accept dict or list input for flexibility
        if isinstance(inputs, dict):
            x = inputs['input_x']
            y = inputs['input_y']
        else:
            x, y = inputs

        admi = self.lstm1(x, training=training)
        pla = self.lstm2(y, training=training)
        out = self.concat([admi, pla])
        output = self.dense(out)
        return output

def my_model_function():
    model = MyModel()
    # Build the model by calling it with dummy data to create weights
    dummy_input = GetInput()
    model(dummy_input)
    # Compile with Adam optimizer and binary crossentropy loss as per example
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def GetInput():
    # Provide an input matching the expected input shapes: two tensors each of shape (batch, 1, 1)
    # Batch size: chosen arbitrary, e.g., 2 to match common batching in example
    batch_size = 2
    x = tf.random.uniform((batch_size, 1, 1), dtype=tf.float32)
    y = tf.random.uniform((batch_size, 1, 1), dtype=tf.float32)
    # Return as dict since model supports dict inputs (also will work with list)
    return {'input_x': x, 'input_y': y}

