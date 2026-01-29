# tf.random.uniform((B, 64), dtype=tf.float32) ← inferred input shape from code: (batch_size, 64)

import tensorflow as tf

class LayerTest(tf.keras.layers.Layer):
    def __init__(self):
        super(LayerTest, self).__init__()

    def call(self, inputs) -> tf.Tensor:
        # inputs is a dictionary with keys like 'step_0', 'step_1'
        predictions = inputs
        
        # Create a new dict to hold normalized tensors to avoid side effects seen in TF 2.3
        predictions2 = {}
        for k in predictions.keys():
            predictions2[k] = tf.math.l2_normalize(predictions[k], axis=-1)

        # Aggregate loss from all normalized steps; here sum the mean of each normalized tensor
        loss = 0.0
        for k in predictions2.keys():
            loss += tf.reduce_mean(predictions2[k])

        return loss


class MyModel(tf.keras.Model):
    def __init__(self, target_dim=64):
        super(MyModel, self).__init__()
        self.target_dim = target_dim
        
        # Input layer will be created by functional Model outside, so just layers here
        # Lambda layers return the input as is, representing the two 'steps'
        self.step_0_layer = tf.keras.layers.Lambda(lambda x: x)
        self.step_1_layer = tf.keras.layers.Lambda(lambda x: x)

        self.layer_test = LayerTest()

    def call(self, inputs):
        # inputs shape: (batch_size, target_dim)
        # Create dictionary predictions as in original code
        predictions = {
            'step_0': self.step_0_layer(inputs),
            'step_1': self.step_1_layer(inputs)
        }

        # Pass dict to custom LayerTest which does l2 normalization and loss aggregation
        output = self.layer_test(predictions)
        return output


def my_model_function():
    # Create a Keras functional model instance wrapping MyModel for convenient use like in the issue

    target_dim = 64
    inputs = tf.keras.Input(shape=(target_dim,), name="input_tensor")
    # Instantiate MyModel and connect the inputs through it
    model_instance = MyModel(target_dim)
    outputs = model_instance(inputs)
    # Wrap in a functional Model to support model.summary() and compile if needed
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def GetInput():
    # Return a random tensor input matching expected shape (batch_size, 64)
    # Using float32 is default for tf.random.uniform
    batch_size = 8  # example batch size
    target_dim = 64
    return tf.random.uniform((batch_size, target_dim), dtype=tf.float32)

