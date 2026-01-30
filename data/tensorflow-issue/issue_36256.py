import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(2,
                                        input_dim=1,
                                        use_bias=True))
model.add(tensorflow.keras.layers.LeakyReLU())
#model.add(tensorflow.keras.layers.ReLU())
model.add(tensorflow.keras.layers.Dense(1,
                                        use_bias=True))

inputs = tensorflow.random.uniform((100,1), 1.0, -1.0)

with tensorflow.GradientTape(watch_accessed_variables=False) as tape:
	tape.watch(model.trainable_weights)
	temp = model(inputs)

jacobian = tape.jacobian(temp, model.trainable_weights)

#model.add(tensorflow.keras.layers.LeakyReLU())
model.add(tensorflow.keras.layers.ReLU())