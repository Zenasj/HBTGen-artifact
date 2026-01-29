# tf.random.uniform((B, 2), dtype=tf.float32)  ← Input shape inferred from example: (batch_size, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example Sequential model from the original issue:
        # Dense(512, relu) -> Dropout(0.5) -> Dense(512, relu) -> Dense(2)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2)
        
        # Note: The original issue used tf.keras.losses.MeanAbsoluteError as a metric by mistake,
        # which causes an error on save/load. Here we avoid using losses as metrics.

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with valid loss and metrics to avoid issues:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]  # Use metrics, not losses here
    )
    return model

def GetInput():
    # Returns a random tensor shaped (batch_size, 2) to match input shape expected by MyModel
    # Use batch_size=32 as a reasonable default
    return tf.random.uniform((32, 2), dtype=tf.float32)

