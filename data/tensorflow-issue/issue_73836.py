import os

import tensorflow as tf
import tf_keras


class ToyModel(tf_keras.Model):
    def __init__(self, embedding_size_bytes: int, var_count: int):
        super().__init__()
        self.embedding_size_bytes = embedding_size_bytes
        self.var_count = var_count
        self.dense = tf_keras.layers.Dense(
            1, activation="sigmoid", kernel_initializer=tf_keras.initializers.GlorotNormal(seed=0)
        )

    def build(self, input_shape: tf.TensorShape):
        self.embeddings: dict[str, tf.Variable] = {}
        vocab_size = int(self.embedding_size_bytes / 256 / 4)
        for i in range(self.var_count):
            name = f"embedding_{i}"
            self.embeddings[name] = self.add_weight(
                name=name,
                shape=(vocab_size, 256),
                initializer=tf_keras.initializers.RandomUniform(seed=i),
                dtype=tf.float32,
                trainable=True,
            )

        self.dense.build(tf.TensorShape((None, 256)))
        self.built = True

    def call(self, inputs: tf.Tensor):
        embedding_values = []
        for i in range(self.var_count):
            name = f"embedding_{i}"
            embedding_values.append(tf.nn.embedding_lookup(self.embeddings[name], tf.squeeze(inputs, axis=1)))

        overall_embedding = tf.reduce_mean(embedding_values, axis=0)
        return self.dense(overall_embedding)


def get_weights_path(name: str):
    return os.path.join(os.path.dirname(__file__), "toy_model_weights", name, "weights")

from typing import Sequence

import os
import shutil
import time
from collections import defaultdict

import tensorflow as tf
from tensorflow.train.experimental import MaxShardSizePolicy, ShardableTensor

from toy_model import ToyModel, get_weights_path, get_model_name


if __name__ == "__main__":
    num_gb = 20
    var_count = 1

    embedding_size_bytes = num_gb * 1024 * 1024 * 1024
    model = ToyModel(embedding_size_bytes, var_count)
    model.build(tf.TensorShape((None, 1)))

    model_name = "test_model"
    weights_path = get_weights_path(model_name)
    shutil.rmtree(os.path.dirname(weights_path), ignore_errors=True)

    shard_size = 2 * 1024 * 1024 * 1024  # 2GB
    sharding_callback = MaxShardSizePolicy(shard_size)

    start_time = time.time()
    model.save_weights(weights_path, options=tf.train.CheckpointOptions(experimental_sharding_callback=sharding_callback))
    print(f"Saving weights took {time.time() - start_time:.2f} seconds")

from typing import Any, Mapping

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.train.experimental import MaxShardSizePolic

from prepare_model import SingleFileMaxShardSizePolicy
from toy_model import ToyModel, get_model_name, get_weights_path


def create_tf_configs(worker_count: int, ps_count: int, include_evaluator: bool):
    """Create TF_CONFIGs for a cluster."""
    cluster_dict: dict[str, list[str]] = {}
    if worker_count:
        cluster_dict["worker"] = [f"localhost:{pick_unused_port()}" for _ in range(worker_count)]
    if ps_count:
        cluster_dict["ps"] = [f"localhost:{pick_unused_port()}" for _ in range(ps_count)]

    cluster_dict["chief"] = [f"localhost:{pick_unused_port()}"]

    tf_configs: list[TFConfig] = []
    for i in range(worker_count):
        tf_configs.append({"cluster": cluster_dict, "task": {"type": "worker", "index": i}})

    for i in range(ps_count):
        tf_configs.append({"cluster": cluster_dict, "task": {"type": "ps", "index": i}})

    if include_evaluator:
        tf_configs.append({"cluster": cluster_dict, "task": {"type": "evaluator", "index": 0}})

    tf_configs.append({"cluster": cluster_dict, "task": {"type": "chief", "index": 0}})

    return tf_configs


def _create_process(tf_config: Mapping[str, Any], log_dir: str):
    name = tf_config["task"]["type"] + "_" + str(tf_config["task"]["index"])
    command = [sys.executable, os.path.basename(__file__)]

    log_file_stdout = os.path.join(log_dir, f"stdout_{name}.log")
    log_file_stderr = os.path.join(log_dir, f"stderr_{name}.log")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file_stdout, "a") as stdout, open(log_file_stderr, "a") as stderr:
        print(f"Starting {name} process...")
        env_ = os.environ.copy() | {"TF_CONFIG": json.dumps(tf_config)}
        return subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env_)


NUM_GB = 20
VAR_COUNT = 1

shard_size = 2 * 1024 * 1024 * 1024  # 2GB
PS_SAVE_SHARD_POLICY = MaxShardSizePolicy(shard_size)


def run():
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

    task_type = resolver.task_type
    print(f"Task type: {task_type}")
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

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        resolver,
        variable_partitioner=tf.distribute.experimental.partitioners.MaxSizePartitioner(
            max_shard_bytes=1 * 1024 * 1024 * 1024
        ),
    )
    print("Building model...")
    model_name = "test_model"
    weights_path = get_weights_path(model_name)

    with strategy.scope(), tempfile.TemporaryDirectory() as temp_dir_name:
        embedding_size_bytes = NUM_GB * 1024 * 1024 * 1024
        model = ToyModel(embedding_size_bytes, VAR_COUNT)
        start_time = time.time()

        model.load_weights(weights_path)
        model.build(tf.TensorShape((None, 1)))
        done_loading_time = time.time()
        print(f"Loading weights took {done_loading_time - start_time:.2f} seconds")
        save_path = os.path.join(temp_dir_name, "weights")
        model.save_weights(
            save_path, options=tf.train.CheckpointOptions(experimental_sharding_callback=PS_SAVE_SHARD_POLICY)
        )
        done_saving_time = time.time()
        print(f"Saving weights took {done_saving_time - done_loading_time:.2f} seconds")
        print("Done")


def main():
    if "TF_CONFIG" in os.environ:
        run()
        return

    tf_configs = create_tf_configs(1, 2, False)
    chief_config = tf_configs[-1]
    model_name = "test_model"
    print(f"Model: {model_name}")
    log_dir = os.path.join(os.path.dirname(__file__), "toy_model_weights", model_name, "logs", "run1")
    shutil.rmtree(log_dir, ignore_errors=True)
    for tf_config in tf_configs[:-1]:
        _create_process(tf_config, log_dir)

    tf.debugging.disable_traceback_filtering()
    os.environ["TF_CONFIG"] = json.dumps(chief_config)
    print("Starting chief")
    run()


if __name__ == "__main__":
    main()