# tf.random.uniform((B, 32), dtype=tf.float32) ← Input shape based on provided dataset shape (batch size variable)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers to match original model:
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return a fresh instance of MyModel.
    # No pretrained weights provided, so initialize fresh.
    model = MyModel()
    # Compile the model to support sparse categorical crossentropy loss and accuracy metric,
    # ensuring correct dtype handling (int32 labels) to avoid the reported TypeError.
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a tuple (inputs, labels) compatible with the model and training loop.
    # Inputs shape: (batch_size, 32), dtype float32
    # Labels shape: (batch_size,), dtype int32 to avoid mismatch errors shown
    batch_size = 32
    inputs = tf.random.uniform(shape=(batch_size, 32), dtype=tf.float32)
    labels = tf.random.uniform(shape=(batch_size,), minval=0, maxval=10, dtype=tf.int32)
    return inputs, labels

