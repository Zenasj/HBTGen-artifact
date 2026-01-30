from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

py
import os
import pprint
import sys
import json
import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.disable_eager_execution()

node_attr = sys.argv[1]
name = node_attr[:-1]
index = node_attr[-1]
os.environ['CUDA_VISIBLE_DEVICES']='-1'

os.environ['TF_CONFIG'] = json.dumps({"cluster": {"worker": ["localhost:5773"], "ps": ["localhost:5711"]}, "task": {"type": name, "index": int(index)}})
strategy = tf.distribute.experimental.ParameterServerStrategy()

# Uncomment below for multi worker mirror strategy.
#os.environ['TF_CONFIG'] = json.dumps({"cluster": {"worker": ["localhost:5773", "localhost:6778"]}, "task": {"type": name, "index": int(index)}})
#strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def input_fn(mode):
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_dataset = (
        datasets['train'] if mode == 'train' else datasets['test']
    )
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    return mnist_dataset.map(scale).cache().repeat(2).shuffle(10000).batch(4)

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

def main():
    tf_config = json.loads(os.environ['TF_CONFIG'])
    job_name = tf_config['task']['type']
    job_index = tf_config['task']['index']

    if job_name == 'ps':
        server = tf.distribute.Server(
            tf_config['cluster'], job_name=job_name, task_index=job_index
        )
        server.join()
    else:
      train_dataset = input_fn('train')
      ckpt_full_path = os.path.join(sys.argv[2], 'model.ckpt-{epoch:04d}')
      callbacks = [
          tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True, verbose=1, save_freq=1),
      ]

      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
      train_datasets_no_auto_shard = train_dataset.with_options(options)

      with strategy.scope():
        model = build_and_compile_cnn_model()
      model.fit(x=train_datasets_no_auto_shard, epochs=3,steps_per_epoch=3, callbacks=callbacks)

if __name__ == '__main__':
    main()