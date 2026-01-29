# tf.random.uniform((1, 1024, 7), dtype=tf.float32) ← Inferred input shape from batch_input_shape in original LSTM layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original: two stacked LSTM layers, stateful, return_sequences=True
        self.lstm1 = tf.keras.layers.LSTM(
            128, 
            return_sequences=True, 
            stateful=True, 
            batch_input_shape=(1, 1024, 7))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(
            128, 
            return_sequences=True, 
            stateful=True)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        # Final Dense layer with 8 outputs and linear activation
        self.dense = tf.keras.layers.Dense(8, activation='linear')

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel.
    # Important: The original model was compiled with the Adam optimizer set as tf.train.AdamOptimizer(0.01)
    # In TF 2.x, Adam uses different API but we keep similar learning rate.
    model = MyModel()
    # Compile the model with Mean Squared Error loss and Adam optimizer with lr=0.01
    # The original used tf.train.AdamOptimizer, replaced by tf.keras.optimizers.Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # batch_size=1, time_steps=1024, features=7 as inferred from batch_input_shape
    # Use float32 dtype which is default for LSTM layers
    return tf.random.uniform((1, 1024, 7), dtype=tf.float32)

