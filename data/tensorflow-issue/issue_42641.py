# tf.random.uniform((1200, 18, 15), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, layer1=32, layer2=32, layer3=16, dropout_rate=0.5, 
                 activation='relu', **kwargs):
        super().__init__(**kwargs)
        # Bidirectional GRU with return_sequences=True to keep time dimension
        self.bidirectional_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(layer1, return_sequences=True))
        # Average pooling along time dimension with pool size 2
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=2)
        # Conv1D layer (extractor)
        self.conv1d = tf.keras.layers.Conv1D(layer2, 3, activation=activation, padding='same', name='extractor')
        # Flatten for feeding dense layers
        self.flatten = tf.keras.layers.Flatten()
        # Dense layers
        self.dense1 = tf.keras.layers.Dense(layer3, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1)  # output single continuous value
        
    def call(self, inputs, training=False):
        # inputs shape expected (batch, timesteps, features) e.g. (1200,18,15)
        x = self.bidirectional_gru(inputs)   # (batch, timesteps, 2*layer1)
        x = self.avg_pool(x)                  # (batch, timesteps//2, 2*layer1)
        x = self.conv1d(x)                   # (batch, timesteps//2, layer2)
        x = self.flatten(x)                  # (batch, timesteps//2 * layer2)
        x = self.dense1(x)                   # (batch, layer3)
        x = self.dropout(x, training=training)
        x = self.dense2(x)                   # (batch, 1)
        return x


def my_model_function():
    # Return a compiled instance of MyModel with default parameters
    model = MyModel()
    # Use default optimizer and loss as described: Adam and mse
    model.compile(optimizer='adam', loss='mse')
    return model


def GetInput():
    # Return a random tensor input matching MyModel input shape: (1200, 18, 15)
    # Use float32 dtype as is standard for TF models
    return tf.random.uniform((1200, 18, 15), dtype=tf.float32)

