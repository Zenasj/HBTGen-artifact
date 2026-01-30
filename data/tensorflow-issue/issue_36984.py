from tensorflow import keras

py
import os
from multiprocessing import Process, Queue

import tensorflow as tf


class Trainable(object):
    def __init__(self):
        self.queue = Queue()
        self.valid_handle = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])  # [1, 2, 3, 4, 5]
        self.train_handle = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def eval(self, q):
        print('Process to write: %s' % os.getpid())
        tmp = -1
        for parsed_record in self.valid_handle:
            print(parsed_record)
            tmp = parsed_record
        self.queue.put((q + 1, tmp))

    def train(self):
        process = None
        print('Process to read: %s' % os.getpid())
        for context in self.train_handle:
            if process:
                valid_detail = self.queue.get()
                process = None
                print(valid_detail)
                if 8 in valid_detail:
                    print('Early stopping')
                    break

            process = Process(target=self.eval, args=(context.numpy(),))
            process.start()
            if not self.queue.empty():
                valid_detail = self.queue.get()
                process = None
                print(valid_detail)


if __name__ == '__main__':
    model = Trainable()
    model.train()

py
import os
from multiprocessing import Process, Queue

import tensorflow as tf

print(tf.version.GIT_VERSION, tf.version.VERSION)
# v1.12.1-25210-gcafd3318ed 2.2.0-dev20200219

fsns_test_file = tf.keras.utils.get_file("fsns.tfrec",
                                         "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")

class Trainable(object):

    def __init__(self):
        self.queue = Queue()
        self.valid_handle = tf.data.TFRecordDataset(filenames=[fsns_test_file])
        self.train_handle = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def eval(self, q):
        print('Process to write: %s' % os.getpid())
        tmp = -1
        for parsed_record in self.valid_handle:
            print(parsed_record)
            tmp = parsed_record
        self.queue.put((q + 1, tmp))

    def train(self):
        process = None
        print('Process to read: %s' % os.getpid())
        for context in self.train_handle:
            if process:
                valid_detail = self.queue.get()
                process = None
                print(valid_detail)
                if 1 in valid_detail:
                    print('Early stopping')
                    break

            process = Process(target=self.eval, args=(context.numpy(),))
            process.start()
            if not self.queue.empty():
                valid_detail = self.queue.get()
                process = None
                print(valid_detail)


if __name__ == '__main__':
    model = Trainable()
    model.train()

def __init__(self):
        self.queue = Queue()
        options = tf.data.Options()
        options.experimental_optimization.autotune = False
        self.valid_handle = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5]).with_options(options)
        self.train_handle = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9])

if __name__ == '__main__':
    set_start_method("spawn", force=True)
    with get_context("spawn").Pool(1) as pool: 
        pool.starmap