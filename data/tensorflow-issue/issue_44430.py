from tensorflow import keras
from tensorflow.keras import layers

call()

model.save()

model.fit()

autograph=True

autograph=False

make_train_function()

model.save()

model.fit()

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.Dense(1)

    def call(self, input):
        x = input
        for i in tf.range(2):
            x = self.layer(x)
        return x

if __name__ == "__main__":
    x = np.arange(10.)
    y = 2 * x

    model = MyModel()
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, epochs=1)
    print("Done training")

    model.save("saved_model")

call

# TODO(mdan): Should the model's call be autographed by default?