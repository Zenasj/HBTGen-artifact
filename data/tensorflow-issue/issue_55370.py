from tensorflow import keras
from tensorflow.keras import layers

# testcase.py
import mlflow
import tensorflow as tf

def init_tf_gpus():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

    tf.config.optimizer.set_jit(True)

    return tf.distribute.TPUStrategy(resolver)

def main():
    strategy = init_tf_gpus()
    with mlflow.start_run(run_name="test"): # disable this line to make it work
        with strategy.scope():
            seq = tf.keras.Sequential([tf.keras.layers.Dense(512)])


if __name__ == "__main__":
    main()