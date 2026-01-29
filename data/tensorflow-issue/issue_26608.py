# tf.random.uniform((16, 5), dtype=tf.float32) ← input shape derived from usage in example with batch size 16 and feature size 5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer with 20 units, ReLU activation, input_shape inferred as (5,)
        self.dense = tf.keras.layers.Dense(20, activation='relu')

    def call(self, inputs, *args, **kwargs):
        # Forward pass:
        # 1. Apply dense layer to input tensor of shape (batch_size, 5)
        x = self.dense(inputs)
        
        # 2. Gather rows indexed by ind = [[0],[1],[2]] from dimension 0
        # This corresponds to selecting the first 3 rows from batch dimension
        ind = [[0], [1], [2]]
        x = tf.gather_nd(x, ind)
        # After gather_nd, x shape is (3, 20)
        
        # 3. Gather rows indexed by ind = [[0],[1]] from the new x
        # This further selects first two rows from the result above
        ind = [[0], [1]]
        x = tf.gather_nd(x, ind)
        # Result shape is (2, 20)
        
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor compatible with MyModel
    # Batch size of 16, feature size of 5, float32 dtype
    return tf.random.uniform((16, 5), dtype=tf.float32)

