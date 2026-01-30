import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow.keras as keras
import numpy as np


def custom_loss(loss_input):
    def loss(y_true, y_pred):
        return y_true * y_pred * loss_input
    return loss


# Create simple model
input_layer = keras.Input(shape=(1,))
loss_input_layer = keras.Input(shape=(1,))

output = keras.layers.Dense(1)(input_layer)

model = keras.Model(inputs=[input_layer, loss_input_layer], outputs=[output])

model.compile(optimizer=keras.optimizers.Adam(lr=1e4), loss=[custom_loss(loss_input_layer)])

# Train the model with some example data
input_data = np.random.random_sample(512)
loss_data = np.random.random_sample(512)
output_data = input_data * -1

model.train_on_batch(x=[input_data, loss_data], y=[output_data])

# Save and load model
model.save("test.h5")
model = keras.models.load_model("test.h5", custom_objects={"loss": custom_loss(loss_input_layer)})

# Test model
model.predict(x=[[np.random.random_sample(1)], [np.random.random_sample(1)]])