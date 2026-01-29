# tf.random.uniform((B, 784), dtype=tf.float32) ← Batch size is dynamic, input is flattened 28x28 grayscale image

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model structure inferred from the keras_model function in the issue
        # Two Conv2D layers with 32 and 64 filters respectively, kernel size 3, stride 2, ReLU activation
        # With dropout 0.5 after each conv
        # Followed by GlobalAvgPool2D and Dense layer with 10 units (softmax)
        self.reshape = tf.keras.layers.Reshape([28, 28, 1], name='input_image')
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', name='cnn0')
        self.dropout1 = tf.keras.layers.Dropout(0.5, name='dropout0')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', name='cnn1')
        self.dropout2 = tf.keras.layers.Dropout(0.5, name='dropout1')
        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D(name='average')
        self.dense = tf.keras.layers.Dense(10, activation='softmax', name='output')

    def call(self, inputs, training=False):
        # inputs expected shape: (batch_size, 784)
        x = self.reshape(inputs)       # (B, 28, 28, 1)
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile to mimic original setup with Adam optimizer and sparse categorical crossentropy
    # Decay argument deprecated in TF 2.x Adam, replaced with learning rate schedules; omitted here
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

def GetInput():
    # Return a random input tensor mimicking MNIST flattened images (batch_size=512 is typical)
    # We keep batch size flexible for real usage; here, let's return batch size 512 as in example
    return tf.random.uniform((512, 784), dtype=tf.float32)

