import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

tf_config = {
    "task": {
        "index": 0,
        "type": "worker"
    },
    "cluster": {
        "worker": ["localhost:21834", "localhost:27271"],
    }
}
os.environ["TF_CONFIG"] = json.dumps(tf_config)
print(json.loads(os.environ["TF_CONFIG"]))


def main(argv):
    del argv  # Unused
    BATCH_SIZE = 100
    SAMPLE_SIZE = 50000
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.AUC()])
    log_dir = FLAGS.logs
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          update_freq='epoch')
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (np.random.randint(1000, size=(SAMPLE_SIZE, 31)),
         np.random.randint(2, size=(SAMPLE_SIZE, 1))))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (np.random.randint(1000, size=(SAMPLE_SIZE, 31)),
         np.random.randint(2, size=(SAMPLE_SIZE, 1))))
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    validation_dataset = validation_dataset.with_options(options)
    model.fit(train_dataset,
              epochs=5,
              steps_per_epoch=10,
              validation_data=validation_dataset,
              validation_steps=5)
    model_dir = FLAGS.logs + '/models'
    model.save(model_dir)

if __name__ == '__main__':
    app.run(main)