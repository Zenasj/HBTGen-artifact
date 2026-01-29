# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Inner model: single Dense layer
        self.inner_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=[1])
        ])
        # Compile the inner model with loss and metrics
        self.inner_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )
        # Construct a Keras Input for outer model forward pass
        self.outer_input = tf.keras.Input(shape=(1,))
        # Get output from inner model
        self.outer_output = self.inner_model(self.outer_input)
        # Outer Model layers wrapper for fit/evaluate compatibility
        self.outer_model = tf.keras.Model(inputs=self.outer_input, outputs=self.outer_output)
        # Compile outer model with own optimizer, loss and metrics - avoid metric name collisions
        self.outer_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError(name="outer_mse")]
        )

    def call(self, inputs):
        # Forward pass through the inner model
        return self.inner_model(inputs)

    def train_step(self, data):
        # Override train_step to delegate to outer_model.train_step to avoid metrics name conflict
        return self.outer_model.train_step(data)

    def test_step(self, data):
        # Override test_step similarly for evaluation
        return self.outer_model.test_step(data)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor shaped (batch_size, 1)
    # Assume batch size 8 for generality
    return tf.random.uniform((8, 1), dtype=tf.float32)

