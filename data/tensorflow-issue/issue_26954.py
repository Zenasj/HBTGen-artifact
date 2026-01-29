# tf.random.uniform((B, 50), dtype=tf.float32) ← Assuming input shape based on example Dense input_shape=(50,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=100, name=None):
        """
        Implements a model with a 'common' backbone plus a final classification layer.

        Args:
          num_classes: Number of classes for the final Dense layer.
          name: Optional name for the model instance.

        This encapsulates the typical transfer learning setup:
          - A common base (backbone) model.
          - A final classification head which can vary in class count.

        This design allows loading weights for the backbone only (excluding the final layer),
        supporting transfer learning where final layers differ in shape.
        """
        super().__init__(name=name)
        # Common backbone layers
        # Here we build a simple example backbone (2 Dense layers)
        # In practice, this could be any model
        self.common = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', name='common_dense_1'),
            tf.keras.layers.Dense(32, activation='relu', name='common_dense_2'),
        ], name='common_model')

        # Final classification layer - variable size depending on the task
        self.final_layer = tf.keras.layers.Dense(num_classes, name='final_layer')

    def call(self, inputs):
        x = self.common(inputs)
        return self.final_layer(x)


def my_model_function(num_classes=100):
    """
    Factory method to create MyModel with specified number of classes.
    """
    return MyModel(num_classes=num_classes)


def GetInput(batch_size=2):
    """
    Returns a random tensor input matching model input shape.
    Assumptions:
      - Input is a batch of 50-dimensional vectors (based on example Dense input_shape=(50,))
      - batch_size can be adjusted as needed.
    """
    return tf.random.uniform((batch_size, 50), dtype=tf.float32)

