# tf.random.uniform((B, 5, 20), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model is a Sequential Conv1D + MaxPooling + Dense stack
        # Input shape: (5, 20)
        # Two Conv1D layers of 32 and 64 filters (kernel_size=1, padding='same'), each followed by MaxPooling1D with pool_size=3 'same' padding
        # Then Flatten and Dense layers: 1024 units relu, and output 1 unit softmax (though softmax on single unit is unusual)
        self.conv1 = tf.keras.layers.Conv1D(32, kernel_size=1, strides=1, activation='relu', padding='same', input_shape=(5, 20))
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=3, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=1, strides=1, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=3, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        # Output activation originally softmax with single unit output - typically should be sigmoid for binary classification
        # But we replicate original for fidelity
        self.dense_out = tf.keras.layers.Dense(1, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense_out(x)
        return x

def my_model_function():
    model = MyModel()
    # Compile with same parameters as original:
    # - optimizer Adam with lr=0.003
    # - loss categorical_crossentropy (though note mismatch with label shape and output shape)
    # - metrics accuracy
    # This replicates exactly the example behaviour from the reported code.
    optimizer = tf.keras.optimizers.Adam(0.003)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

def GetInput():
    # According to the original data generator:
    # samples shape = (5, 20), dtype float32 assumed from np.random.rand
    # So random tensor with shape (batch_size=32, 5, 20)
    # The batch size is 32 * num_gpu in the original code;
    # We pick 32 as a default here.
    return tf.random.uniform((32, 5, 20), dtype=tf.float32)

