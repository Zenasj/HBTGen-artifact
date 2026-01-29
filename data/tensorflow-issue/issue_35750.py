# tf.random.uniform((B,)) ← Input shape unknown from issue; infer scalar or vector labels input for loss demo

import tensorflow as tf

# Since the original issue revolves around a functional API model with a named output "logit_layer"
# and using a custom loss keyed by that name, the inference here is:
# - Inputs and output shape are not explicit, so we assume a simple vector input.
# - The "logit_layer" is the final output Dense layer.
# - The loss function is a softmax cross-entropy wrapped in a custom function.
#
# We wrap that into a single Model subclass "MyModel".
# Given the issue context is loading/saving with custom loss keyed by output layer name,
# this example sets up a matching model and loss dictionary keyed by output name.
#
# The input shape is assumed (e.g., (?, 10)), with 10 classes in logits for classification.
# Output name is set to "logit_layer" following the issue description.
#
# Since the original issue doesn't provide exact input shapes or the architecture,
# here is a minimal reproducible model based on the description.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Example input shape: 20 features
        # Dense layers following the reported pattern
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        # output layer named "logit_layer" for consistency with the issue loss key
        self.logit_layer = tf.keras.layers.Dense(10, name="logit_layer")

    def call(self, inputs):
        x = self.dense1(inputs)
        logits = self.logit_layer(x)
        return logits

def my_loss(labels, logits):
    # Replicate the custom loss wrapper shown in the issue:
    # Use tf.nn.softmax_cross_entropy_with_logits with named outputs
    # Warning: the original TF function expects logits and labels of same shape.
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def my_model_function():
    # Create instance of MyModel
    model = MyModel()
    # Compile the model with the custom loss keyed by the output layer name.
    # This matches the reported pattern causing the unknown keys issue on load.
    # Optimizer and metrics are placeholders for completeness.
    model.compile(
        optimizer='adam',
        loss={'logit_layer': my_loss},
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return input tensor to match MyModel input
    # Assumed shape: batch of 4, each with 20 features
    return tf.random.uniform((4, 20), dtype=tf.float32)

# Notes:
# - This code reflects the pattern from the issue where custom loss keyed by output layer name
#   causes errors on loading the saved model.
# - To fully replicate the problem and save/load cycle, the model would be trained and saved,
#   then loaded back providing the custom loss function in custom_objects.
# - The custom loss dictionary keyed by output layer name 'logit_layer' is exactly as in the issue.
#
# This minimal example provides a fully functional MyModel class, a custom loss that can be compiled,
# and an input generator to run inference or training steps.

