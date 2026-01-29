# tf.random.uniform((10, 3, 5), dtype=tf.float32) ← inferred from input_shape=(3, len(train_X[0])) with batch_size=10 and train_X originally shaped (200,5)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Stateful LSTM layer with 86 units, input shape (3 timesteps, 5 features), batch size 10
        self.lstm = tf.keras.layers.LSTM(86, return_sequences=True, stateful=True,
                                         batch_input_shape=(10, 3, 5))
        # Dense layer outputs 2 classes (categorical)
        self.dense = tf.keras.layers.Dense(2)
        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)
        x = self.dense(x)
        x = self.activation(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return input tensor compatible with the model expected shape:
    # batch_size=10, time_steps=3, features=5 (as per the example data shape)
    # Use float32 dtype as typical for TF inputs
    return tf.random.uniform((10, 3, 5), dtype=tf.float32)

