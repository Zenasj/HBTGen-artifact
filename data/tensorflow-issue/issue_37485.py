# tf.random.uniform((B, timesteps, features), dtype=tf.float32) ← Input shape inferred from issue context where LSTM input_shape=(timesteps, features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, timesteps=10, features=8, num_classes=5):
        super().__init__()
        # Inferred default values for timesteps, features, and num_classes since not explicitly given in issue.

        # Encoder LSTM layers
        self.lstm1 = tf.keras.layers.LSTM(32, activation='relu',
                                          return_sequences=True,
                                          input_shape=(timesteps, features))
        self.lstm2 = tf.keras.layers.LSTM(16, activation='relu', return_sequences=False)
        # RepeatVector to match input timesteps
        self.repeat = tf.keras.layers.RepeatVector(timesteps)
        # Decoder LSTM layers
        self.lstm3 = tf.keras.layers.LSTM(16, activation='relu', return_sequences=True)
        self.lstm4 = tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)
        # Final time-distributed dense with softmax for classification per timestep
        self.time_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_classes, activation='softmax')
        )

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.repeat(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        output = self.time_dense(x)
        return output


def my_model_function():
    # Create an instance of MyModel with default params
    model = MyModel()
    # Compile as in the original issue to mimic intended usage
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return input tensor matching model's input shape: (batch_size, timesteps, features)
    # Use batch size of 4 for example
    batch_size = 4
    timesteps = 10    # Must match model default
    features = 8      # Must match model default
    # Since model output uses categorical crossentropy with softmax, input could be float32 features
    return tf.random.uniform((batch_size, timesteps, features), dtype=tf.float32)

