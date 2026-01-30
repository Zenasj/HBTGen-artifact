import random
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import (
    float32,
    function,
    TensorSpec,
)
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TestModel(Model):
    def __init__(
        self
    ):
        super(TestModel, self).__init__()

        self.dense = Dense(100)

    @function(
        input_signature=[
            TensorSpec([None, 512], float32, name="responses"),
            TensorSpec([None, 512], float32, name="contexts"),
        ]
    )
    def call(
        self,
        responses,
        contexts
    ):
        return self.dense(responses + contexts)

model = TestModel()
x = tf.random.normal((1, 512))
_ = model(x, x)
tf.saved_model.save(model, "directory")