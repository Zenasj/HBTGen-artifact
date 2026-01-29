# tf.random.uniform((B, 3), dtype=tf.float32) ← From sample input shape (5,3) in issue code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(4)
        
    def call(self, inputs, training=False):
        # The key issue described in the original reported bug relates to the `training` parameter being
        # passed as None instead of True or False during training vs inference. Here, we accept `training`
        # as boolean or bool tensor correctly, allowing behavior depending on mode.
        # For demonstration, just pass training to layers that might behave differently in train/infer mode.
        return self.dense(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel.
    # No extra weights or special initialization needed beyond standard Keras initialization.
    return MyModel()

def GetInput():
    # Create random input tensor matching shape expected by MyModel: (batch_size, 3)
    # The original example used batch = 5. We use batch=5 here for consistency.
    # Data type float32 matches the example.
    return tf.random.uniform((5, 3), dtype=tf.float32)

