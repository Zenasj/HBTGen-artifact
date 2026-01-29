# tf.random.uniform((2000, 992, 5), dtype=tf.float64) ← input shape inferred from the issue: (batch_size=2000, backward=992, number_features=5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original Sequential model in the issue:
        # Input shape: (992, 5)
        # LSTM with 40 units, return_sequences=False
        # Dropout with 0.2
        # Dense output of 3 units with sigmoid activation
        self.lstm = tf.keras.layers.LSTM(40, return_sequences=False)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(3, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dropout(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()

    # Build the model by calling it on a sample input for weights initialization
    sample_input = GetInput()
    _ = model(sample_input, training=False)

    # Compile the model similarly to the issue snippet
    model.compile(optimizer='adam', loss='mse')
    return model

def GetInput():
    # Return a random tensor input with shape (2000, 992, 5) matching batch_size, backward, number_features
    # dtype = tf.float64 to match the original dataset dtypes in the code
    return tf.random.uniform((2000, 992, 5), dtype=tf.float64)

