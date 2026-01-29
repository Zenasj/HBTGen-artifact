# tf.random.uniform((B, 1440*3), dtype=tf.float32)  ← Input shape inferred from `input_shape=(num_inputs,)` where num_inputs=1440*3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        num_inputs = 1440 * 3
        n_hidden = 2 * num_inputs
        # Define a 6-layer fully connected network as per the issue's example:
        # 1 input Dense layer + 4 hidden Dense layers + 1 output Dense layer
        self.dense_input = tf.keras.layers.Dense(n_hidden, activation=tf.nn.relu, input_shape=(num_inputs,))
        self.hidden_layers = [tf.keras.layers.Dense(n_hidden, activation=tf.nn.relu) for _ in range(4)]
        self.output_layer = tf.keras.layers.Dense(2)  # num_outputs=2

    def call(self, inputs, training=False):
        # Forward pass through the model replicating the example code
        x = self.dense_input(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Returns an instance of MyModel
    model = MyModel()
    # Compile the model similarly to the original example:
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model

def GetInput():
    # Generates a random tensor with shape matching the input of MyModel
    # The original input shape: (batch_size, 1440*3).
    # Batch size inferred from example set to a reasonable default (e.g. 512)
    batch_size = 512
    input_shape = (batch_size, 1440 * 3)
    # Generate random float32 tensor similar to train_X (np.random.randn)
    # Use a normal distribution to approximate original input distribution
    return tf.random.normal(input_shape, dtype=tf.float32)

