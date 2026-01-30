from tensorflow import keras

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return 2 * inputs

@tf.function
def step_model(model, inputs):
    return model(inputs)

@tf.function
def step(inputs):
    return model(inputs)

inputs = tf.convert_to_tensor(1, dtype=tf.float32)
model = MyModel()
# This works as expected
print(f"step() = {step(inputs)}") # 2.0
print(f"step() concrete functions: {step._list_all_concrete_functions_for_serialization()}") # [<tensorflow.python.eager.function.ConcreteFunction object at 0x13a2c0510>]
# This does not, no concrete function is saved
print(f"step_model() = {step_model(model, inputs)}") # 2.0
print(f"step_model() concrete functions: {step_model._list_all_concrete_functions_for_serialization()}") # []