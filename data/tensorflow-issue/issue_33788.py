# tf.random.uniform((64, 256, 256, 3), dtype=tf.float32) ← Assumed batch size 64, input image size 256x256 with 3 color channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the architecture from the original Sequential model,
        # converted to subclassing with tf.keras layers.
        self.conv2d = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3), activation='relu', input_shape=(256,256,3)
        )
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3,3))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(units=4, activation='softmax')

    def call(self, inputs, training=False):
        # Implement the forward pass.
        x = self.conv2d(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)  # dropout behaves differently in training/inference
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and compile the model to closely match the original example
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Generate a random batch of input images with shape matching model input
    # Batch size 64 like in the original code
    return tf.random.uniform((64, 256, 256, 3), dtype=tf.float32)

