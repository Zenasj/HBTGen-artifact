import random
from tensorflow import keras

import os
import numpy as np
import tensorflow as tf
from absl import app


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.x = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[1]))

    def call(self, inputs):
        return inputs


def main(argv):
    del argv
    tf.random.set_seed(123)
    net = Net()
    optim = tf.optimizers.Adam(learning_rate=0.001)

    dataset = tf.data.Dataset.from_tensor_slices((np.arange(10)))
    dataset = dataset.shuffle(10, reshuffle_each_iteration=True)
    dataset = dataset.batch(2)
    db_iter = iter(dataset)
    step = tf.Variable(0)
    checkpoint = tf.train.Checkpoint(
        step=step, optimizer=optim, net=net, db_iter=db_iter
    )
    manager = tf.train.CheckpointManager(checkpoint, "./ckpts", max_to_keep=50)
    manager.restore_or_initialize()
    print(step)
    for epoch in range(10):
        manager.restore_or_initialize()
        for _ in range(dataset.cardinality()):
            batch = next(db_iter)
            step.assign_add(1)
            print(batch)
        db_iter = iter(dataset)
        manager.save()
        print(f"Epoch {epoch} finished.")


if __name__ == "__main__":
    app.run(main)

### Relevant log output

### CODE