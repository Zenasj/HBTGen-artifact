# tf.random.uniform((B, 10), dtype=tf.float32)  # inferred input shape from the example with shape=(10,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the inner model: Input(10) => Dense(10, name='inner_layer')
        inner_input = tf.keras.layers.Input(shape=(10,))
        x = tf.keras.layers.Dense(10, name='inner_layer')(inner_input)
        self.inner_model = tf.keras.Model(inner_input, x, name='inner_model')

        # Define the outer model, which uses inner_model as a layer:
        # Input(10) => Dense(10, name='outer_layer') => inner_model
        outer_input = tf.keras.layers.Input(shape=(10,))
        y = tf.keras.layers.Dense(10, name='outer_layer')(outer_input)
        y = self.inner_model(y)
        self.outer_model = tf.keras.Model(outer_input, y, name='outer_model')

        # Define an "extended inner model" that appends a Dense(10) layer after inner_layer inside inner_model
        inner_layer_output = self.inner_model.get_layer('inner_layer').output
        extended_inner_output = tf.keras.layers.Dense(10, name='extended_inner_dense')(inner_layer_output)
        self.extended_inner_model = tf.keras.Model(self.inner_model.input, extended_inner_output)

        # Attempt to define an "extended inner model" but with outer_input as input
        # That tries to get inner_layer output through the outer_model nested structure.
        # This reflects the problem in the question.
        # It works for inner_model, but not for outer_model directly by layer.output
        outer_inner_layer_output = self.outer_model.get_layer('inner_model').get_layer('inner_layer').output
        extended_outer_inner_output = tf.keras.layers.Dense(10, name='extended_outer_inner_dense')(outer_inner_layer_output)
        # We capture the ValueError that the issue raises by commenting out model creation here.
        # Instead, we build a Lambda layer that simply calls outer_model and pipes to the extended_inner_model
        # This is a workaround for the nested model layer output issue.

        # Build a functional model manually: Input -> outer_model -> access inner_layer functional output
        # Note: Cannot directly create model with outer_input -> extended_outer_inner_output because of graph disconnect.

        # Instead, we implement a forward pass logic in `call` that uses the nested models to produce outputs
        # The call method will return both:
        #   - The output of outer_model
        #   - The output from extended_inner_model applied to the intermediate inner_layer output

        # It's not trivial to get the intermediate tensor from nested model in Functional API outside inner_model,
        # because of graph disconnected errors, so we use the inner_model separately here.

    def call(self, inputs):
        # Forward pass through outer model's first Dense layer
        x = self.outer_model.get_layer('outer_layer')(inputs)

        # Forward through inner_model
        inner_out = self.inner_model(x)

        # Also forward through extended_inner_model (adds extra Dense layer after inner_layer)
        # We get the 'inner_layer' output first by manually applying layer on x
        inner_layer_layer = self.inner_model.get_layer('inner_layer')
        intermediate_inner_layer_out = inner_layer_layer(x)
        extended_inner_out = self.extended_inner_model(intermediate_inner_layer_out)

        # Output a tuple of (outer_model output, extended_inner_model's output)
        # to represent the nested model outputs and an extended output.
        return (inner_out, extended_inner_out)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input shape for outer_model input is (10,)
    # Batch size can be any positive integer; choose 2 as an example
    return tf.random.uniform((2, 10), dtype=tf.float32)

