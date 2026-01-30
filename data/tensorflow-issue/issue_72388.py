from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import json
import os


if bool(int(os.environ.get("TF_USE_LEGACY_KERAS", "0"))):
    import tf_keras as keras
else:
    import tensorflow.keras as keras


print("TensorFlow version:", tf.__version__)


mnist = keras.datasets.mnist


def get_address_and_port():
    return "127.0.0.1", find_free_port()


def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _setup_tensorflow_environment(worker_addresses, index: int):
    """Set up distributed Tensorflow training information.
    This function should be called on each worker.
    Args:
        worker_addresses: Addresses of all the workers.
        index: Index (i.e. world rank) of the current worker.
    """
    config = {
        "cluster": {"worker": worker_addresses},
        "task": {"type": "worker", "index": index},
    }
    os.environ["TF_CONFIG"] = json.dumps(config)


def train_fn(worker_addresses=["127.0.0.1:12345"], index=0):
    _setup_tensorflow_environment(worker_addresses, index)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation="relu"),
                # NOTE: This errors for some reason when running with either MP or ray
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10),
            ]
        )
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    predictions = model(x_train[:1]).numpy()
    predictions

    tf.nn.softmax(predictions).numpy()

    loss_fn(y_train[:1], predictions).numpy()

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)

    probability_model = keras.Sequential([model, keras.layers.Softmax()])

    probability_model(x_test[:5])


def get_url():
    address, port = get_address_and_port()
    return f"{address}:{port}"


def run_with_ray():
    import ray

    train_fn_task = ray.remote(train_fn)

    num_workers = 2
    worker_addresses = [get_url() for _ in range(num_workers)]

    ray.get([train_fn_task.remote(worker_addresses, i) for i in range(num_workers)])


def run_with_mp():
    from multiprocessing import Process

    num_workers = 2
    worker_addresses = [get_url() for _ in range(num_workers)]

    p1 = Process(target=train_fn, args=(worker_addresses, 0))
    p2 = Process(target=train_fn, args=(worker_addresses, 1))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == "__main__":
    import sys

    runner = sys.argv[1]
    if runner == "vanilla":
        train_fn()
    elif runner == "ray":
        run_with_ray()
    elif runner == "mp":
        run_with_mp()