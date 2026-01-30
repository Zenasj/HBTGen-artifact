from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name='MyScheduler')
class MyScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, **kwargs):
        super(MyScheduler, self).__init__(**kwargs)

    def __call__(self, step):
        return step

    def get_config(self):
        return {}


inputs = tf.keras.Input(10)
outputs = tf.keras.layers.Dense(10)(inputs)
model = tf.keras.Model(inputs, outputs)

lr_scheduler = MyScheduler()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
model.compile(optimizer=optimizer, loss="mse")


def get_dataset(repeat):
    inputs_data = tf.ones([16, 10])
    labels_data = tf.ones([16, 10])
    dataset = (
        tf.data.Dataset.from_tensors(inputs_data)
        .map(
            lambda x: (
                inputs_data,
                labels_data,
                None,
            )
        ).repeat(repeat)
    )
    return dataset


model.fit(get_dataset(3), epochs=1)
print(model.optimizer.iterations, lr_scheduler(model.optimizer.iterations))

path = "./foo/"
model.save(path)
loaded = tf.keras.models.load_model(path)
loaded.fit(get_dataset(4), epochs=1)
print(loaded.optimizer.iterations, lr_scheduler(loaded.optimizer.iterations))