# tf.random.uniform((B, 784), dtype=tf.float32) ← inferred input shape from MNIST flattened examples (batch size B)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple feedforward network similar to the MNIST example:
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(784,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')
        
        # We will keep track of a custom metric tensor for demonstration (like the example)
        # but tf.keras Models do not expose metrics_tensors attribute, so we define metrics explicitly.
        self.custom_metrics = []

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        
        # Example of computing a custom "loss-like" metric based on layer outputs.
        # This mimics the pattern in the issue where they computed tf.reduce_mean(layer.output).
        # Here we just add mean activation metrics of each dense layer output for demo.
        self.custom_metrics = []  # reset custom metrics per call
        
        for layer_output in [x, output]:
            # mean activation over batch and neurons, keepdims=True for scalar with shape (1,)
            metric = tf.reduce_mean(layer_output, keepdims=True)
            self.custom_metrics.append(metric)
        
        return output

    def compute_metrics(self):
        # Return a list of custom metrics (tensors) for monitoring, e.g., in training logs
        return self.custom_metrics

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching (batch_size, 784) for flattened MNIST images
    # Assuming batch size 32 as default
    return tf.random.uniform((32, 784), dtype=tf.float32)

