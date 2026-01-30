from tensorflow import keras
from tensorflow.keras import models

import keras
import tensorflow as tf

class MyModel(keras.Model):
    def __init__(self):
        super().__init__(autocast=False)

    def call(self, inputs):
        return 0

def main():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = MyModel()

    inputs = tf.zeros(shape=(1, 1), dtype=tf.float32)

    prediction_0 = model(inputs)

    model_path = "test_model"

    model.save(model_path)

    loaded_model = keras.models.load_model(model_path)

    prediction_1 = loaded_model(inputs)

if __name__ == "__main__":
    main()