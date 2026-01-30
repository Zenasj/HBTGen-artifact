import random
from tensorflow.keras import optimizers

from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

image_input = keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = keras.Input(shape=(None, 10), name='ts_input')

x1 = layers.Conv2D(3, 3)(image_input)
x1pool = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2pool = layers.GlobalMaxPooling1D(name='x2pool')(x2)

x = layers.concatenate([x1pool, x2pool])

# score_output = layers.Dense(1, name='score_output')(x)
class_output = layers.Dense(5, activation='softmax', name='class_output')(x)

# original outputs from docs example code
# outputs = [class_output, score_output]

# the commented one works (only by coincidence!), but the uncommented one doesnt
# outputs = [class_output, x2pool]
outputs = [x2pool, class_output]

model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss={'class_output': keras.losses.CategoricalCrossentropy()})


img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))


# we supply the targets for the loss that we tagged to the class_output head.
train_dataset = tf.data.Dataset.from_tensor_slices(
    ({'img_input': img_data, 'ts_input': ts_data},
     {'class_output': class_targets}))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model.fit(train_dataset, epochs=3)