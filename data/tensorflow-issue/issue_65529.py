# tf.random.uniform((B, 20), dtype=tf.float32)  # Inferred input shape from sklearn make_classification with 20 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model replicating the Sequential model from the issue:
        # Single Dense layer with 2 units and softmax activation.
        self.dense = tf.keras.layers.Dense(units=2, activation='softmax')
        
    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # For demonstration, compile the model to match the reported compilation in the issue.
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Generate a random input tensor matching the expected input: batches of 20-dimensional vectors.
    # We'll assume batch size 8 as a reasonable default.
    batch_size = 8
    input_shape = (batch_size, 20)
    return tf.random.uniform(input_shape, dtype=tf.float32)

