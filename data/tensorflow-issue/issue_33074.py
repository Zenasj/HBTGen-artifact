# tf.random.uniform((B, 96, 72, 1), dtype=tf.float64)  ← Input shape inferred from model InputLayer in issue (height=96, width=72, channels=1)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model architecture as described:
        # Input shape is (96, 72, 1)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=4, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(18, activation='softmax')

        # Note: The InputLayer is implicitly defined by input shape in call

    def call(self, inputs, training=False):
        # The model expects inputs with shape (batch, 96, 72, 1)
        # We explicitly check this shape and raise error if mismatch,
        # mimicking eager execution behavior described in the issue.
        input_shape = tf.shape(inputs)
        static_shape = inputs.shape  # static shape or None

        # The input shape should be (batch, 96, 72, 1)
        # We check last 3 dims shape
        expected_shape = (96, 72, 1)

        # Using static shape check first if available
        if static_shape.rank == 4:
            if static_shape[1] != expected_shape[0] or static_shape[2] != expected_shape[1] or static_shape[3] != expected_shape[2]:
                raise ValueError(f'Input tensor has shape {static_shape[1:]} but expected {expected_shape}')
        else:
            # dynamic shape check in graph mode
            cond1 = tf.equal(input_shape[1], expected_shape[0])
            cond2 = tf.equal(input_shape[2], expected_shape[1])
            cond3 = tf.equal(input_shape[3], expected_shape[2])
            all_ok = tf.logical_and(tf.logical_and(cond1, cond2), cond3)
            # Raise an error in graph mode by tf.assert or tf.debugging.assert
            with tf.control_dependencies([tf.debugging.assert_equal(all_ok, True, message=f"Input shape mismatch. Expected last three dims {expected_shape}, got {input_shape[1:]}")]):
                inputs = tf.identity(inputs)

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel, ready for usage
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input shape expected by MyModel
    # Use dtype tf.float64 as dataset was with float64 in original issue
    # Batch size assumed arbitrary; let's use batch size = 16 as in dataset.batch(16)
    batch_size = 16
    height = 96
    width = 72
    channels = 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float64)

