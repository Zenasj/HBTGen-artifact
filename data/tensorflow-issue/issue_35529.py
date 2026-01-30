from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow.keras as keras
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = keras.layers.Embedding(51,100)
        self.layer = keras.layers.Dense(51)

    def call(self,x):
        x = self.emb(x)
        x = self.layer(x)
        return x


strategy = tf.distribute.MirroredStrategy()

data = [[i for i in range(random.randint(10,50))] for j in range(400)]


def iterator():
    for i in range(len(data)):
        yield data[i], data[i]


with strategy.scope():
    model = Model()
    optimizer = keras.optimizers.Adam()

dataset = tf.data.Dataset.from_generator(iterator, output_types=(tf.int64, tf.int64))
batchfier = dataset.padded_batch(4, padded_shapes=([None], [None]))
batchfier = strategy.experimental_distribute_dataset(batchfier)


@tf.function(input_signature=batchfier.element_spec)
def multi_gpu_step(x,y):
    def example_update_step(x, y):
        with tf.GradientTape() as tape:
            y_ = model(x)
            batch_loss = keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_, from_logits=True)
            losses = batch_loss / strategy.num_replicas_in_sync
        step_grad = tape.gradient(losses, model.trainable_variables)
        optimizer.apply_gradients(zip(step_grad, model.trainable_variables))
        return tf.reduce_mean(batch_loss,1)
    example_loss = strategy.experimental_run_v2(
        example_update_step, args=(x, y))
    losses_sum = strategy.reduce(
        tf.distribute.ReduceOp.SUM, example_loss, axis=0)
    return losses_sum


for x,y in batchfier:
    multi_gpu_step(x,y)