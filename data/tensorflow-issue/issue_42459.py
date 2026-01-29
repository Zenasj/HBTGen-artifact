# tf.random.uniform((4, 2), dtype=tf.float32) ← Input is batch of 4 samples, each with 2 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense output with 2 units and softmax activation as in the example
        self.dense = tf.keras.layers.Dense(units=2, activation='softmax', name='output')

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel, compiled with the workaround that explicitly uses sparse_categorical_accuracy
    model = MyModel()
    # Compile with Adam optimizer; learning rate fixed from code example (note lr param is deprecated but kept for faithfulness)
    optimizer = tf.keras.optimizers.Adam(lr=10)
    # Use loss and metrics matching the issue's minimal example and workaround
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

def GetInput():
    # Return a random tensor matching the example's dummy_data_x shape and type
    # Four samples, each sample has 2 features (float32)
    return tf.random.uniform((4, 2), dtype=tf.float32)

