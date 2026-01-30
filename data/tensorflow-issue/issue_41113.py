from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras

def get_dataset():
    x = tf.zeros([10], dtype=tf.float32)
    x = tf.data.Dataset.from_tensors(x)

    y = tf.constant([5])
    y = tf.data.Dataset.from_tensor_slices(y)

    dataset = tf.data.Dataset.zip((x, y))
    dataset = dataset.batch(1)
    dataset = dataset.repeat()
    return dataset

def main():
    # NOTE: You must set os.environ["TF_CONFIG"] as appropriate
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
    # assert strategy.num_replicas_in_sync == 2

    # Create dataset
    dataset = get_dataset()

    with strategy.scope():
        # Construct model
        model = keras.Sequential(
            layers=[
                tf.keras.layers.experimental.SyncBatchNormalization(),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
        )

    model.fit(x=dataset, steps_per_epoch=10 ** 6, epochs=10 ** 3)


if __name__ == "__main__":
    main()