# tf.random.uniform((B, 28*28), dtype=tf.float32) ← inferred input shape based on the examples in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Defining layers similar to the example Sequential model,
        # but using functional style to avoid issues with InputLayer and Sequential
        self.dense1 = tf.keras.layers.Dense(300, activation='relu', input_shape=(28*28,))
        self.dense2 = tf.keras.layers.Dense(100, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel similar to the compiled Keras model in the issue
    model = MyModel()
    # Compile here to mimic the original example (optional since training not included)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # batch size B = 1 is chosen as default
    return tf.random.uniform((1, 28*28), dtype=tf.float32)

