from tensorflow import keras
from tensorflow.keras import layers

# Use aforeshared code to define setup_dataset
import numpy as np

train, valid, inp_voc_size, tar_voc_size = setup_dataset()
np.save('train.npy', [(x.numpy(), y.numpy()) for x, y in train])
np.save('valid.npy', [(x.numpy(), y.numpy()) for x, y in valid])

# I also ran commands to get the constants and note them somewhere.
# In the second run, I therefore hard-code them for simplicity.
# input vocab size is 8443, target vocab size is 8356
# train set comprises 704 batches, validation set has 17

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

def reload_dataset(path):
    """Reload a dumped dataset and finish formatting it."""
    data = np.load(path, allow_pickle=True).tolist() 
    def generator(): 
        for inputs, target in data: 
            yield ((inputs, target[:, :-1]), target[:, 1:]) 
    types = ((tf.int64, tf.int64), tf.int64) 
    shape = (((None, None), (None, None)), (None, None)) 
    dataset = tf.data.Dataset.from_generator(generator, types, shape) 
    return dataset 


# use aforeshared code to define setup_model


def main():
    train = reload_dataset('train.npy')
    valid = reload_dataset('valid.npy')
    model = setup_model(8443, 8356)
    model.fit(
        epochs=10, x=train.repeat(), steps_per_epoch=704,
        validation_data=valid.repeat(), validation_steps=17,
    )

if __name__ == '__main__':
    main()

class OneHotEmbedding(tf.keras.layers.Embedding):
    "Embedding layer with one-hot dot-product retrieval mechanism."""

    def call(self, inputs):
        """Embed some inputs."""
        one_hot = tf.one_hot(inputs, depth=self.input_dim, dtype=tf.float32)
        return tf.keras.backend.dot(one_hot, self.embeddings)