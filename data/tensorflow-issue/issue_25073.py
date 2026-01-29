# tf.random.uniform((1, 10), dtype=tf.float32)  # inferred input shape from the reproduction example: single batch, 10 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize layers with specified weight initializers and biases as in the original Sequential model:
        self.dense1 = tf.keras.layers.Dense(
            25, input_dim=10,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=1),
            bias_initializer='zeros',
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            25,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2),
            bias_initializer='zeros',
            activation='relu'
        )
        self.dense3 = tf.keras.layers.Dense(
            10,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=3),
            bias_initializer='zeros',
            activation='softmax'
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Return an instance of MyModel with a default SGD optimizer (lr=0.1) and categorical_crossentropy loss compiled.
    model = MyModel()
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def GetInput():
    # Return input tensor matching the input shape expected by the model
    # The original example input was numpy array shape (1, 10) with dtype float.
    # Use uniform random float tensor with same shape and dtype float32.
    return tf.random.uniform((1, 10), dtype=tf.float32)

