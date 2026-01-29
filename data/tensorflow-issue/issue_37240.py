# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is unknown from the issue, so assuming a generic 4D tensor as example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model implements a demonstration of how tf.keras.losses.get can be used
        # to obtain losses either from string identifiers or dict config objects,
        # illustrating the subtle distinction between function-based and class-based losses.

        # Submodule representing the loss obtained by string identifier "categorical_crossentropy"
        # This returns a function loss, supervised externally.
        self.loss_fn_from_string = tf.keras.losses.get("categorical_crossentropy")

        # Submodule representing the loss obtained by dict config for the class-based loss,
        # equivalent to: CategoricalCrossentropy(from_logits=True)
        loss_config = {"class_name":"CategoricalCrossentropy", 
                       "config":{"from_logits":True}}
        self.loss_instance_from_config = tf.keras.losses.get(loss_config)

    def call(self, inputs):
        # For demonstration, inputs is assumed to be a tuple of (y_true, y_pred)
        # to be compatible with losses.

        y_true, y_pred = inputs

        # Compute loss using function-based loss (string identifier variant)
        # This should behave like: categorical_crossentropy(y_true, y_pred)
        loss1 = self.loss_fn_from_string(y_true, y_pred)

        # Compute loss using class-based loss instance with `from_logits=True`
        loss2 = self.loss_instance_from_config(y_true, y_pred)

        # Now compare the two losses numerically and return a boolean tensor
        # indicating whether they are close within a tolerance.
        # This showcases a fusion/fork between the two methods described.
        comparison = tf.math.less_equal(tf.abs(loss1 - loss2), 1e-5)

        # Return a dict with all results for introspection
        return {
            "loss_fn_from_string": loss1,
            "loss_instance_from_config": loss2,
            "close": comparison,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Loss functions expect as input a tuple of (y_true, y_pred),
    # both tensors of same shape.
    # Typical shape for classification: (batch_size, num_classes)
    # Assuming batch size 2 and 5 classes (arbitrary).
    batch_size = 2
    num_classes = 5

    # y_true is generally one-hot or integer labels.
    # categorical_crossentropy expects y_true as one-hot encoded for logits.
    y_true = tf.random.uniform((batch_size, num_classes), minval=0, maxval=2, dtype=tf.int32)
    y_true = tf.cast(y_true > 0, tf.float32)  # Convert to 0/1 float tensor simulating one-hot presence

    # y_pred is logits or probabilities depending on loss config
    # Here, since `from_logits=True`, y_pred should be raw logits.
    y_pred = tf.random.uniform((batch_size, num_classes), minval=-3, maxval=3, dtype=tf.float32)

    return (y_true, y_pred)

