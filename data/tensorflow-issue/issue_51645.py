from tensorflow.keras import layers
from tensorflow.keras import models

# Third Party
import numpy as np
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.python.keras.models import Model

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.con = Concatenate()
        self.dense = Dense(10, activation="relu")

    def call(self, x):
        x = self.con([x, x])
        return self.dense(x)

if __name__ == "__main__":
   model = MyModel()
   print(model.layers)

from keras.layers import Concatenate, Dense
from keras.models import Model