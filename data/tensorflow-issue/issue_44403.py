from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os
import shutil


class MyLookup(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table_init = tf.lookup.KeyValueTensorInitializer(
            key_dtype=tf.int64,
            keys=[0, 1, 2],
            value_dtype=tf.string,
            values=["A", "B", "C"],
            name="table_init")
        self.index_to_kw = tf.lookup.StaticHashTable(self.table_init, "?")

    def call(self, inputs, **kwargs):
        return self.index_to_kw.lookup(inputs)


class TestSaveProblem(tf.test.TestCase):

    def determine_and_clear_test_workdir(self):
        testname = self.id()
        result = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "tmp_test_workdir", testname))
        shutil.rmtree(result, ignore_errors=True)
        return result

    def testSaveProblem(self):
        export_dir = self.determine_and_clear_test_workdir() + "/saved_model"

        exampledata = [1, 2]

        input = tf.keras.layers.Input(shape=1, dtype=tf.int64)
        output = MyLookup(name='result')(input)
        model = tf.keras.Model(inputs=[input], outputs=[output])

        # save and load
        model.save(export_dir, save_format='tf', include_optimizer=False)
        loaded_model = tf.saved_model.load(export_dir, [tf.saved_model.SERVING]).signatures[
            'serving_default']

        # test after saving and loading (works!)
        loaded_result = loaded_model(tf.constant(exampledata, dtype=tf.int64))['result']
        self.assertAllEqual([b"B", b"C"], loaded_result)

if __name__ == '__main__':
    tf.test.main()