# tf.random.uniform((B=1, H=224, W=224, C=3), dtype=tf.float32) ← inferred from Input layer shape (224, 224, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent to Sequential model from issue example
        # Layers: Input(224,224,3) -> Conv2D(256,3x3,l2) -> GAP -> Dense(10,sigmoid)
        self.conv = tf.keras.layers.Conv2D(
            256, 
            (3, 3), 
            kernel_regularizer=tf.keras.regularizers.l2()
        )
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(10, activation='sigmoid')
        
        # We track accuracy metric here as an example metric
        # This metric is computed during call if labels are passed separately
        # Note: In the issue context, metrics are only active when loss is set during compile,
        # but here we demonstrate a way to support metric calculation manually.
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

    def call(self, inputs, training=False, labels=None):
        """
        Forward pass through the network.
        
        Args:
          inputs: Tensor with shape (batch, 224, 224, 3)
          labels: Optional. Ground truth tensor with shape (batch, 10)
                  Used only to update accuracy metric inside the call.
        
        Returns:
          Output predictions tensor of shape (batch, 10)
        """
        x = self.conv(inputs)
        x = self.gap(x)
        out = self.dense(x)
        
        # If labels provided, update accuracy metric state
        if labels is not None:
            self.accuracy_metric.update_state(labels, out)
        
        return out

    def reset_metrics(self):
        # Reset metric states, useful between epochs/training steps
        self.accuracy_metric.reset_states()
    
    def get_metrics(self):
        # Return current result of accuracy metric, or None if no updates yet
        return self.accuracy_metric.result()

def my_model_function():
    # Returns an instance of MyModel, ready for compile/use
    return MyModel()

def GetInput():
    # Generate a random input tensor matching model input shape: (batch, 224, 224, 3)
    # Using batch size 1 as default
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

