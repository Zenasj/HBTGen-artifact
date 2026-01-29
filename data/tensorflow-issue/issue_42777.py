# tf.random.uniform((100, 784), dtype=tf.float32) ← input shape inferred from MNIST example with batch size 100 and input size 784

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        nb_inputs = 784
        hidden_units = [32, 32]
        nb_classes = 10
        
        # Backbone model: a small MLP with two dense layers
        self.backbone = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")
                for i, units in enumerate(hidden_units)
            ],
            name="backbone",
        )
        
        # Final classifier head with softmax activation
        self.classifier = tf.keras.layers.Dense(nb_classes, activation="softmax", name="output")

    def call(self, inputs, training=False):
        """
        Forward pass through backbone and classifier.

        Args:
            inputs: Tensor of shape (batch_size, 784), dtype float32
            training: boolean flag for training mode

        Returns:
            Tensor of shape (batch_size, 10) - class probabilities
        """
        x = self.backbone(inputs, training=training)
        output = self.classifier(x)
        return output

def my_model_function():
    # Return an instance of MyModel with initialized weights
    model = MyModel()
    # Build the model explicitly to ensure it is "built" before saving/loading
    # This avoids issues related to unbuilt layers (the error from the issue)
    dummy_input = tf.random.uniform((1, 784), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Generate a random input tensor consistent with the expected input
    # shape (batch, 784), dtype float32, batch size here assumed 100 for demonstration
    return tf.random.uniform((100, 784), dtype=tf.float32)

