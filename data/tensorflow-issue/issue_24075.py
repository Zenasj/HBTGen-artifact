# tf.random.uniform((B=..., C=1, H=28, W=28), dtype=tf.float16) ← input shape inferred as channels_first (1,28,28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Model as described in the issue, channels_first data format
        self.conv1 = tf.keras.layers.Conv2D(
            filters=35, kernel_size=(3,3), strides=(1,1), padding='same',
            activation='relu', data_format='channels_first',
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(0.01),
            kernel_initializer='glorot_normal',
            input_shape=(1, 28, 28)  # input_shape here is for Keras build; not strictly required for subclassing
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same', data_format='channels_first')

        self.conv2 = tf.keras.layers.Conv2D(
            filters=36, kernel_size=(3,3), strides=(1,1), padding='same',
            activation='relu', data_format='channels_first',
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(0.01),
            kernel_initializer='glorot_normal'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same', data_format='channels_first')

        self.conv3 = tf.keras.layers.Conv2D(
            filters=36, kernel_size=(3,3), strides=(1,1), padding='same',
            activation='relu', data_format='channels_first',
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(0.01),
            kernel_initializer='glorot_normal'
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same', data_format='channels_first')

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(576, activation='relu')
        # Output layer with 10 units and relu activation as per original code.
        # Note: Usually for classification one would use softmax or linear with logits.
        self.dense2 = tf.keras.layers.Dense(10, activation='relu')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)     # Expect input shape (B, 1, 28, 28)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # The model expects input shape with data_format='channels_first': (B, C=1, H=28, W=28)
    # Use float16 dtype as per the dataset conversion in the example
    # Batch size: choose a reasonable batch size, say 32 for demonstration
    B = 32
    C = 1
    H = 28
    W = 28
    # Return a random tensor input
    return tf.random.uniform(shape=(B, C, H, W), dtype=tf.float16)

