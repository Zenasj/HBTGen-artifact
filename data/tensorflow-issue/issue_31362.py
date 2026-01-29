# tf.random.uniform((B, H, W, C), dtype=...) ← Input shape and dtype unknown from issue; assuming shape=(1,) and dtype=tf.float32 for demonstration

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Inferred: since original issue was about side effects from non-TF code inside a tf.function,
        # here we create a simple passthrough layer to emulate "some TF computation"
        self.dummy_layer = tf.keras.layers.Lambda(lambda x: x)  # identity layer for placeholder

    def call(self, inputs, obj):
        # inputs: some tensor input
        # obj: custom python object with attribute .value

        # Perform some TF computation on inputs (here identity for example)
        x = self.dummy_layer(inputs)

        # Emulate the original problem: modifying obj.value inside tf.function
        # This modification will only effectively happen on first tracing
        # Here we replicate exactly what happens in the original issue to illustrate behavior

        # Note: obj.value += 1 is a Python side effect that only runs during tracing
        # So we add it here to replicate the behavior that it only increments once on first run
        obj.value += 1

        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a tensor input compatible with MyModel
    # Issue didn't specify input details; we assume a 1D float32 tensor of shape (1,)
    return tf.random.uniform((1,), dtype=tf.float32)

