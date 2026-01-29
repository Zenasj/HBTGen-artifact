# tf.random.uniform((4, 5, 500, 500, 3), dtype=tf.float32) for the image input batch
# tf.random.uniform((4, 105000, 1), dtype=tf.float32) for the audio input batch

import tensorflow as tf

class ActionModel(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        # Simple CNN to process image input (shape: (5, 500, 500, 3))
        # Assuming input_shape = (5, 500, 500, 3)
        # We'll treat 5 as a temporal or sequence dimension and combine conv and pooling accordingly
        self.conv1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'), input_shape=input_shape
        )
        self.pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2))
        self.conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
        self.pool2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2))
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.dense = tf.keras.layers.Dense(64, activation='relu')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        # x shape: (batch_size, 5, features)
        # We'll reduce across the sequence dimension (5) by averaging
        x = tf.reduce_mean(x, axis=1)
        x = self.dense(x)
        return x

class AudioAndFinalModel(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        # Simple 1D CNN on audio input (shape: (105000, 1))
        self.conv1 = tf.keras.layers.Conv1D(32, 8, strides=4, activation='relu', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling1D(4)
        self.conv2 = tf.keras.layers.Conv1D(64, 8, strides=4, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling1D(4)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(64, activation='relu')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize submodels with proper input shapes:
        # image input shape: (5, 500, 500, 3)
        # audio input shape: (105000, 1)
        self.action_model = ActionModel(input_shape=(5, 500, 500, 3))
        self.audio_model = AudioAndFinalModel(input_shape=(105000,1))

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        # Output 2 units with tanh activation for binary classification as in original code
        self.out_layer = tf.keras.layers.Dense(2, activation='tanh')

    def call(self, inputs, training=False):
        # inputs: tuple(image_input, audio_input)
        image_input, audio_input = inputs
        x1 = self.action_model(image_input)
        x2 = self.audio_model(audio_input)
        combined = self.concat([x1, x2])
        z = self.dense1(combined)
        z = self.dropout(z, training=training)
        out = self.out_layer(z)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 4 (from BATCH_SIZE in issue)
    batch_size = 4

    # Image input shape: (5, 500, 500, 3)
    image_shape = (batch_size, 5, 500, 500, 3)
    # Audio input shape: (105000, 1)
    audio_shape = (batch_size, 105000, 1)

    # Generate random input tensors with float32 dtype
    image_input = tf.random.uniform(image_shape, dtype=tf.float32)
    audio_input = tf.random.uniform(audio_shape, dtype=tf.float32)

    return (image_input, audio_input)

