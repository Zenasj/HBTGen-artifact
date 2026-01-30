from typing import Generator

import tensorflow as tf


class SomeObj:
    def __init__(self, x):
        self.x = x

    def __del__(self):
        print("__del__ called!")


def build_dataset(
    obj: SomeObj,
) -> tf.data.Dataset:
    def _generator() -> Generator[tf.Tensor, None, None]:
        while True:
            yield tf.convert_to_tensor(obj.x, dtype=tf.float32)

    return tf.data.Dataset.from_generator(
        _generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )


def train_for_a_fold(fold):
    dataset = build_dataset(SomeObj(fold))
    for x in dataset.take(3):
        print(x)


def main():
    train_for_a_fold(0)
    train_for_a_fold(1)
    train_for_a_fold(2)


if __name__ == "__main__":
    main()