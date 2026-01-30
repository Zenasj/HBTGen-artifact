import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

class ThreeLayerMLP(keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.pred_layer = layers.Dense(10, name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)


def main(argv):
    del argv  # Unused args
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    print('Number of devices: %d' % strategy.num_replicas_in_sync)

    with strategy.scope():
        model = model = ThreeLayerMLP(name='3_layer_mlp')
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.RMSprop())

    log_dir = FLAGS.logs
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          update_freq='batch')

    np.random.seed(0)
    x_train, y_train = (np.random.random(
        (60000, 784)), np.random.randint(10, size=(60000, 1)))
    x_test, y_test = (np.random.random(
        (10000, 784)), np.random.randint(10, size=(10000, 1)))

    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE)


    model.fit(
        train_dataset,
        epochs=5,
        steps_per_epoch=10,
        callbacks=tensorboard_callback)

    model_dir = FLAGS.logs + '/models/' + str(task_index)
    model.save(model_dir)


if __name__ == '__main__':
    app.run(main)