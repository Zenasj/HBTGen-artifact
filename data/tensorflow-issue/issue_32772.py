# tf.random.uniform((32, 100), dtype=tf.float32) ← Input shape inferred from the example Gen output

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Single dense layer reducing features to dimension 4
        self.dense = tf.keras.layers.Dense(4)

    def call(self, inputs, training=False):
        # As per the reported issue, training param can be None or boolean, ensure it is boolean
        if training is None:
            training = False  # Fallback to False if None, matching expected behavior
        # Print statement from original example to observe training flag (commented out for clean forward pass)
        # print('Training', training)
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate input batch of shape (32, 100) with float32 values matching example data (np.ones replaced by uniform for variability)
    return tf.random.uniform((32, 100), dtype=tf.float32)

