import random
from tensorflow import keras
from tensorflow.keras import models

import subprocess
import tensorflow as tf

inputs = {
    "first_feature": tf.constant([1, 2, 3]),
    "second_feature": tf.constant([4, 5, 6]),
}

class MyModel(tf.keras.Model):
    def call(self, inputs, training=None):
        return tf.random.uniform([3, 1])

model = MyModel()
model.compile(optimizer="sgd", loss="mse")
model.fit(inputs, tf.constant([7, 8, 9]))
model.save("model")
dict_model_def = subprocess.check_output("saved_model_cli show --dir model --tag_set serve --signature_def serving_default".split())
print("MODEL")
print(dict_model_def.decode("utf-8"))

print("RESTORED MODEL")
restored_model = tf.keras.models.load_model("model")
restored_model.save("restored_model")
restored_model_def = subprocess.check_output("saved_model_cli show --dir restored_model --tag_set serve --signature_def serving_default".split())
print(restored_model_def.decode("utf-8"))

(({'first_feature': TensorSpec(shape=(3,), dtype=tf.int32, name='inputs/first_feature'),
   'second_feature': TensorSpec(shape=(3,), dtype=tf.int32, name='inputs/second_feature')},
  None),
 {})