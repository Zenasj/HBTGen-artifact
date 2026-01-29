# tf.random.uniform((B, 3), dtype=tf.float64) ← Input shape inferred from sklearn linnerud dataset features (3 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Layers as per original FullyConnectedNetwork
        # Note: LSTM expects 3D inputs (batch, timesteps, features), original code feeds 2D input; we assume timesteps=1 for LSTM input to work.
        # The issue code does not specify shape changes, so we add expand dims before LSTM in call.

        self.layer1 = tf.keras.layers.Dense(9)  # input_shape=(3,) will be inferred later
        self.layer2 = tf.keras.layers.LSTM(8, return_sequences=True)
        self.layer3 = tf.keras.layers.Dense(27)
        self.layer4 = tf.keras.layers.Dropout(0.5)
        self.layer5 = tf.keras.layers.Dense(27)
        self.layer6 = tf.keras.layers.Concatenate()
        self.layer7 = tf.keras.layers.Dense(3)

    def call(self, x, training=False):
        # x: shape (batch_size, 3)
        x1 = tf.nn.tanh(self.layer1(x))               # (batch_size, 9)
        # For LSTM, add a time dimension: (batch_size, timesteps=1, features=9)
        y = self.layer2(tf.expand_dims(x1, axis=1))   # (batch_size, 1, 8)
        # Remove time dimension for next layers to match concatenation (x and y shapes)
        y = tf.squeeze(y, axis=1)                      # (batch_size, 8)

        x = tf.nn.selu(self.layer3(x1))                # (batch_size, 27)
        x = self.layer4(x, training=training)          # dropout active only if training=True
        x = tf.nn.relu(self.layer5(x))                  # (batch_size, 27)

        # Concatenate x (27) and y (8): (batch_size, 35)
        x = self.layer6([x, y])

        x = self.layer7(x)                              # (batch_size, 3)
        return x


def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weights to load, just a fresh model.
    return MyModel()


def GetInput():
    # The original data X has shape (samples, 3)
    # The dtype in user code was set to float64 due to tf.keras.backend.set_floatx('float64')
    # To avoid dtype mismatch issues, ensure input dtype is float32 as modern TF prefers float32 for operations.
    # The model layers default to float32 dtype in TF 2.x, so input should be float32.

    batch_size = 4  # As in original batching from the issue
    feature_dim = 3

    # Generate a random float32 tensor input matching shape (batch_size, 3)
    # Values arbitrary between 0 and 1 to mimic normalized input
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

