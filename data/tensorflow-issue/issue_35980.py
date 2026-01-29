# tf.random.uniform((1, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST example in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the MNIST simple model as per the official beginner tutorial snippet
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel, compiled with settings matching the example
    model = MyModel()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor input compatible with MyModel: MNIST images normalized to [0,1]
    # Shape: (batch=1, height=28, width=28)
    return tf.random.uniform((1, 28, 28), minval=0, maxval=1, dtype=tf.float32)

