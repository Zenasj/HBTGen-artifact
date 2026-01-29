# tf.random.uniform((B, 64, 256), dtype=tf.float32), tf.random.uniform((B, 512), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer W1 processes input1 of shape (64,256) to output (64,512)
        self.W1 = tf.keras.layers.Dense(512)
        # Dense layer W2 processes input2 - after expansion - to match shape (64,512)
        self.W2 = tf.keras.layers.Dense(512)
        
    def call(self, inputs):
        x1_, x2_ = inputs  # x1_: (B,64,256), x2_: (B,512)
        
        # Apply Dense on x1_ with shape (B,64,256) -> (B,64,512)
        x1 = self.W1(x1_)
        
        # Expand dims on x2_ from (B,512) to (B,1,512) to allow broadcasting along dim=1
        x2_expanded = tf.expand_dims(x2_, axis=1)  # (B,1,512)
        
        # Apply Dense on x2_expanded (broadcasted per time-step)
        x2 = self.W2(x2_expanded)  # (B,1,512)
        
        # Broadcast add to sum (B,64,512) + (B,1,512) -> (B,64,512)
        y = tf.math.add(x1, x2)
        
        return y

def my_model_function():
    # Return an instance of MyModel (weights are randomly initialized by default)
    return MyModel()

def GetInput():
    # Generate random tensor inputs matching the model input shapes:
    # Input 1: (B, 64, 256), Input 2: (B, 512)
    batch_size = 1
    x1 = tf.random.uniform((batch_size, 64, 256), dtype=tf.float32)
    x2 = tf.random.uniform((batch_size, 512), dtype=tf.float32)
    return (x1, x2)

