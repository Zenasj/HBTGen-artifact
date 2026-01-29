# tf.random.uniform((B, 32), dtype=tf.float32) ← Inferred input shape: (batch_size, 32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the resolved guidance, the first Dense layer specifies input_shape=(32,)
        # Activations 'relu' for hidden layers; 'softmax' for output.
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(32,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Create an instance of MyModel and compile it similarly to the example
    model = MyModel()
    # Using Adam optimizer with learning rate 0.001 as in original code sample
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random float32 tensor matching the expected input shape (batch, 32)
    # Batch size is arbitrary; 32 chosen here for consistency with example batch size
    batch_size = 32
    # Use tf.random.uniform with dtype float32 to avoid dtype mismatch errors
    return tf.random.uniform(shape=(batch_size, 32), dtype=tf.float32)

