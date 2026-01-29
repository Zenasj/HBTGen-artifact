# tf.random.uniform((16, 3), dtype=tf.float32) ← Input matches shape (batch=16, features=3)

import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # Print whether running eagerly during call to demonstrate eager execution setting
        tf.print('Running eagerly: ', tf.executing_eagerly())
        return inputs

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)
        self.custom_layer = CustomLayer()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.custom_layer(x)
        return x

    def compile(self, optimizer='rmsprop', loss=None, metrics=None,
                loss_weights=None, sample_weight_mode=None,
                weighted_metrics=None, run_eagerly=None, **kwargs):
        # Fix the run_eagerly attribute assignment bug described:
        # In TF version around 2.2.0 early dev, run_eagerly added as explicit arg,
        # but original code fetched run_eagerly from kwargs by mistake.
        # Here, we correctly set self.run_eagerly attribute.
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        sample_weight_mode=sample_weight_mode,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        **kwargs)
        # Explicitly assign model.run_eagerly to the argument value:
        if run_eagerly is not None:
            self.run_eagerly = run_eagerly

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Produce a random float32 tensor with shape (16, 3), matching example batch size & features
    return tf.random.uniform((16, 3), dtype=tf.float32)

