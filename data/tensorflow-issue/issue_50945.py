# tf.random.uniform((512, None, None, None), dtype=tf.float32) ← Input shape inferred from issue context: (timesteps, features) not explicit, assume (None, feature_dim)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CuDNNLSTM is no longer explicitly available as a separate class
        # In TF 2.x, tf.keras.layers.LSTM with default parameters 
        # uses the fused CuDNN kernels automatically when GPU is available
        # We implement the equivalent model using tf.keras.layers.LSTM with proper config
        
        # According to original model:
        # First LSTM layer with 256 units, return_sequences=True, input_shape=(timesteps, features)
        # Second LSTM Layer with 256 units, return_sequences=False
        # Dropout layers after each LSTM with rate 0.2
        # Dense layer with 1 unit + tanh activation
        
        # Assumptions:
        # - input shape is (timesteps, features) with features unknown, so parameterized
        # - batch size is variable, not fixed
        # - We use tf.keras.layers.LSTM (which wraps CuDNN kernels under the hood on GPU)
        # - Use activation='tanh' internally for LSTM cells (default)
        
        self.lstm1 = tf.keras.layers.LSTM(
            256,
            return_sequences=True,
            recurrent_activation='sigmoid',  # default to CuDNN compatible activation
            name="lstm1"
        )
        self.dropout1 = tf.keras.layers.Dropout(0.2, name="dropout1")
        
        self.lstm2 = tf.keras.layers.LSTM(
            256,
            return_sequences=False,
            recurrent_activation='sigmoid',
            name="lstm2"
        )
        self.dropout2 = tf.keras.layers.Dropout(0.2, name="dropout2")
        
        self.dense = tf.keras.layers.Dense(1, name="dense")
        self.activation = tf.keras.layers.Activation('tanh', name="tanh_activation")
        
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.dense(x)
        x = self.activation(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Provide a random input with shape (batch_size, timesteps, features)
    # Assumptions:
    # - batch_size = 512 (as per original training batch size)
    # - timesteps = 100 (arbitrary example since original not given)
    # - features = 64 (arbitrary example)
    batch_size = 512
    timesteps = 100
    features = 64
    return tf.random.uniform((batch_size, timesteps, features), dtype=tf.float32)

