import tensorflow as tf
from tensorflow import keras

def gen(epoch_number):
    yield epoch_number # Just representative. In reality this will be some function of epoch_number and data.

dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.int32), args=[x])

# A custom keras model derived from tf.keras.Model
model = SomeModel()

model.fit(dataset)