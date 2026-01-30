from tensorflow import keras
from tensorflow.keras import layers

from typing import Any

import os
import tempfile

import tensorflow as tf
from tensorflow.python.distribute.sharded_variable import ShardedVariable


# More realistic example this would be done by Parameter Server Strategy.
def shard_variables_creator(partitioner) -> Any:
    def _creator(next_creator, **kwargs):
        var = next_creator(**kwargs)
        if var.shape.rank == 0:
            return var

        num_shards = partitioner(var.shape, var.dtype, axis=0)
        if num_shards[0] == 1:
            return var

        shard_count = num_shards[0]
        shards = []
        start = 0
        for index in range(shard_count):
            shard_name = var.name.removesuffix(":0") + f"/part_{index}"
            size = var.shape[0] // shard_count + (1 if var.shape[0] % shard_count > index else 0)
            shards.append(tf.Variable(var[start : start + size], name=shard_name))
            start += size

        return ShardedVariable(shards)

    return _creator


partitioner = tf.distribute.experimental.partitioners.MaxSizePartitioner(max_shard_bytes=100 * 16 * 4)

with tf.variable_creator_scope(shard_variables_creator(partitioner)):
    toy_model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(100, 32), tf.keras.layers.Dense(1, activation="sigmoid")]
    )
    toy_model.compile(loss="binary_crossentropy", optimizer=tf.optimizers.experimental.Adam())
    toy_model.build(input_shape=(None, 1))
    toy_model.optimizer.build(toy_model.trainable_variables)  # type: ignore

temp_dir = tempfile.gettempdir()
weights_path = os.path.join(temp_dir, "model_weights")
toy_model.save_weights(weights_path)

toy_model2 = tf.keras.Sequential([tf.keras.layers.Embedding(100, 32), tf.keras.layers.Dense(1, activation="sigmoid")])
toy_model2.compile(loss="binary_crossentropy", optimizer=tf.optimizers.experimental.Adam())
toy_model2.build(input_shape=(None, 1))
toy_model2.optimizer.build(toy_model2.trainable_variables)  # type: ignore
toy_model2.load_weights(weights_path)

with strategy.scope():
  toy_model2 = tf.keras.Sequential([tf.keras.layers.Embedding(100, 32), tf.keras.layers.Dense(1, activation="sigmoid")])
  toy_model2.compile(loss="binary_crossentropy", optimizer=tf.optimizers.experimental.Adam())
  toy_model2.build(input_shape=(None, 1))
  toy_model2.optimizer.build(toy_model2.trainable_variables)  # type: ignore
  toy_model2.load_weights(weights_path)

from typing import Any, Mapping

import json
import os
import tempfile
from multiprocessing import Process

import tensorflow as tf

from portpicker import pick_unused_port

__spec__ = None


def create_tf_configs(worker_count: int, ps_count: int):
    """Create TF_CONFIGs for a cluster."""
    cluster_dict: dict[str, list[str]] = {}
    if worker_count:
        cluster_dict["worker"] = [f"localhost:{pick_unused_port()}" for _ in range(worker_count)]
    if ps_count:
        cluster_dict["ps"] = [f"localhost:{pick_unused_port()}" for _ in range(ps_count)]

    cluster_dict["chief"] = [f"localhost:{pick_unused_port()}"]

    tf_configs = []
    for i in range(worker_count):
        tf_configs.append({"cluster": cluster_dict, "task": {"type": "worker", "index": i}})

    for i in range(ps_count):
        tf_configs.append({"cluster": cluster_dict, "task": {"type": "ps", "index": i}})

    tf_configs.append({"cluster": cluster_dict, "task": {"type": "chief", "index": 0}})

    return tf_configs


def _create_process(tf_config: Mapping[str, Any]):
    name = tf_config["task"]["type"] + "_" + str(tf_config["task"]["index"])

    print(f"Starting {name} process...")
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    p = Process(target=run)
    p.start()


def run():
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

    task_type = resolver.task_type
    if task_type in ("worker", "ps"):
        print("Starting server...")
        server = tf.distribute.Server(
            resolver.cluster_spec(),
            job_name=resolver.task_type,
            task_index=resolver.task_id,
            protocol=resolver.rpc_layer,
            start=True,
        )
        server.join()

    partitioner = tf.distribute.experimental.partitioners.MaxSizePartitioner(max_shard_bytes=100 * 16 * 4)
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver=resolver, variable_partitioner=partitioner
    )

    print("Building model...")
    with strategy.scope():
        toy_model = tf.keras.Sequential(
            [tf.keras.layers.Embedding(100, 32), tf.keras.layers.Dense(1, activation="sigmoid")]
        )
        toy_model.compile(loss="binary_crossentropy", optimizer=tf.optimizers.experimental.Adam())
        toy_model.build(input_shape=(None, 1))
        toy_model.optimizer.build(toy_model.trainable_variables)  # type: ignore

    print("Saving weights...")
    temp_dir = tempfile.gettempdir()
    weights_path = os.path.join(temp_dir, "model_weights")
    toy_model.save_weights(weights_path)

    # No partitioner used for second ps strategy.
    strategy2 = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver=resolver)
    with strategy2.scope():
        toy_model2 = tf.keras.Sequential(
            [tf.keras.layers.Embedding(100, 32), tf.keras.layers.Dense(1, activation="sigmoid")]
        )
        toy_model2.compile(loss="binary_crossentropy", optimizer=tf.optimizers.experimental.Adam())
        toy_model2.build(input_shape=(None, 1))
        toy_model2.optimizer.build(toy_model2.trainable_variables)  # type: ignore
        print("Loading weights...")
        toy_model2.load_weights(weights_path).assert_consumed()

    print("Done!")


def main():
    tf_configs = create_tf_configs(2, 1)
    chief_config = tf_configs[-1]
    for tf_config in tf_configs[:-1]:
        _create_process(tf_config)

    os.environ["TF_CONFIG"] = json.dumps(chief_config)
    run()


if __name__ == "__main__":
    main()