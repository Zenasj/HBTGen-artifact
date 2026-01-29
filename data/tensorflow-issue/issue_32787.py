# tf.random.uniform((B, 784), dtype=tf.float32) ← input shape for MNIST flattened images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model according to the MNIST CNN example shared in the issue
        self.reshape = tf.keras.layers.Reshape((28, 28, 1), name='input_image')
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), activation='relu', name='cnn0')
        self.dropout1 = tf.keras.layers.Dropout(0.5, name='dropout0')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), activation='relu', name='cnn1')
        self.dropout2 = tf.keras.layers.Dropout(0.5, name='dropout1')
        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D(name='average')
        self.dense = tf.keras.layers.Dense(10, activation='softmax', name='output')

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        x = self.global_avg_pool(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile configuration here is optional if used as Keras model directly
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Generate a batch of random flattened MNIST-like images with batch size 32
    batch_size = 32
    # MNIST images are 28x28 grayscale => 784 flattened inputs
    return tf.random.uniform((batch_size, 784), minval=0, maxval=1, dtype=tf.float32)

