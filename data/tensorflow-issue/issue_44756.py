# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape inferred as 4D float tensor (batch, height, width, channels) as typical for Keras models.
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # For demonstration, we create a simple Conv2D + GlobalAveragePooling2D + Dense model.
        # This is inferred since the original issue relates to TF Keras metric usage,
        # so we include a simple forward pass with something to measure.
        self.conv = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu')
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # Include a metric instance as part of the model to reflect the original issue context.
        # Using tf.keras.metrics.Mean as an example metric from the issue,
        # which was related to garbage collection / memory leaks.
        self.metric = tf.keras.metrics.Mean(name='mean_metric')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        output = self.dense(x)
        
        # Update the metric with the batch outputs.
        self.metric.update_state(output)
        
        # Return the model output and the current metric result for transparency.
        # This reflects the original issue context where metrics are involved in the forward pass.
        metric_result = self.metric.result()
        return output, metric_result

def my_model_function():
    # Returns an instance of MyModel.
    return MyModel()

def GetInput():
    # Returns a random input tensor compatible with MyModel's expected input.
    # Typical image-like input: batch=8, height=64, width=64, channels=3.
    # dtype must be float32 to match conv layer and metric dtype.
    return tf.random.uniform((8, 64, 64, 3), dtype=tf.float32)

