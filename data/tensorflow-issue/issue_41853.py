# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape inferred from issue data: inputs have shape (3,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple dense layer modeling as in the issue example (input shape (3,), output shape (2,))
        self.dense = tf.keras.layers.Dense(2)
    
    def call(self, inputs):
        return self.dense(inputs)
    
    def test_step(self, data):
        # Custom test_step method similar to what tf.keras.Model.test_on_batch does internally
        x, y = data
        y_pred = self(x, training=False)
        loss = tf.keras.losses.mean_squared_error(y, y_pred)
        return tf.reduce_mean(loss)

def my_model_function():
    # Return a compiled instance of MyModel as per example with mean squared error loss and SGD optimizer
    model = MyModel()
    model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
    return model

def GetInput():
    # Return a batched dataset matching the issue's inputs and labels formats
    # Shape: batch size 2 for iteration example, inputs shape (3,), labels shape (2,)
    input_vals = tf.random.uniform((2, 3), dtype=tf.float32)
    label_vals = tf.random.uniform((2, 2), dtype=tf.float32)
    return (input_vals, label_vals)

