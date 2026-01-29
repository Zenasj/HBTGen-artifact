# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape (batch_size, 10 features)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Expect inputs as a tuple: (features, sample_weights)
        However, sample_weights here are only inputs, not used internally in this model since 
        sklearn roc_auc_score cannot be directly used with tensors in a tf.function.
        
        This model replicates the original MLP structure from issue.
        """
        x, w = inputs
        x = self.dense1(x)
        out = self.dense2(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random float input tensor matching features shape (batch_size, 10)
    # Also create a sample_weight tensor of shape (batch_size, 1) filled with random float values
    batch_size = 32  # arbitrary batch size for input
    features = tf.random.uniform((batch_size, 10), dtype=tf.float32)
    sample_weights = tf.random.uniform((batch_size, 1), dtype=tf.float32)
    return (features, sample_weights)

