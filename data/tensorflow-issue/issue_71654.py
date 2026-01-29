# tf.random.uniform((64, 20, 20), dtype=tf.float32) ← batch_size=64, sequence_length=20, features=20
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original Sequential model layers and parameters:
        # - LSTM layers with 100 units, return_sequences=True for first LSTM
        # - unroll=True for LSTM (forces unrolling, useful for debugging)
        # - use_bias=False, recurrent_activation='sigmoid' as specified
        # - Dropout layers with rate 0.2
        # - Final Dense layer with 1 unit (for regression output)
        
        self.lstm1 = tf.keras.layers.LSTM(
            100, return_sequences=True, unroll=True, use_bias=False,
            recurrent_activation='sigmoid'
        )
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        
        self.lstm2 = tf.keras.layers.LSTM(
            100, unroll=True, use_bias=False,
            recurrent_activation='sigmoid'
        )
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    
    # Note: weights are uninitialized here; user should call model.compile and fit
    # according to their use case, e.g., model.compile(optimizer='adam', loss='mse')
    return model

def GetInput():
    # Return random input tensor to match model input shape
    # According to original code: input_shape=(20,20), batch_size=64 is typical
    return tf.random.uniform((64, 20, 20), dtype=tf.float32)

