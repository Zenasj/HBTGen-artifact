import random

import pickle
import tensorflow as tf
from tensorflow.python.training.tracking.tracking import AutoTrackable


class MyTrackable(AutoTrackable):
    def __init__(self):
        self.random_op = {
            'tf.trandom.uniform(())': tf.random.uniform(())
        }


def main():
    my_trackable = MyTrackable()
    value = tf.Session().run(my_trackable.random_op)
    print(type(value))
    pickle.dumps(value)


if __name__ == '__main__':
    main()