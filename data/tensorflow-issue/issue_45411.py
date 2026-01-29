# tf.random.uniform((B, 5, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.n_steps = 5
        # LSTM layer with 200 units and relu activation
        self.lstm = tf.keras.layers.LSTM(200, activation='relu', input_shape=(self.n_steps, 1))
        # Dense layers as per original model
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(50, activation='relu')
        self.out = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        # inputs shape: (batch, n_steps, 1)
        x = self.lstm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x

def my_model_function():
    # Create and compile the model as in the original code
    model = MyModel()
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
    return model

def GetInput():
    # Return a random tensor input matching model's expected input shape:
    # Shape = (batch_size, n_steps, features=1)
    # Choosing batch_size=32 as per the original batch size
    batch_size = 32
    n_steps = 5
    # Random float32 tensor in [0,1)
    return tf.random.uniform((batch_size, n_steps, 1), dtype=tf.float32)

