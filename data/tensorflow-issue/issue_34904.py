import numpy as np
import random
import tensorflow as tf

class FramesDS:

    def example_frames_ds(self, max_len=None):
        def gen():
            example_count = 0
            while True:
                example_id = np.random.randint(low=np.iinfo(np.int64).min,
                                               high=np.iinfo(np.int64).max, dtype='int64')
                frame_count = np.random.randint(low=2, high=10)
                print("{:4d}| {}: frame_count:{}".format(example_count, example_id, frame_count))
                example_count += 1

                max_seq_len = 8
                frames  = np.random.randint(low=0, high=9, size=(frame_count, max_seq_len))
                example_ids = [example_id] * frame_count
                yield example_ids, frames
        ds = tf.data.Dataset.from_generator(gen,
                                            output_types=(tf.int64, tf.int32),
                                            output_shapes=(tf.TensorShape([None, ]), tf.TensorShape([None, None])))
        if max_len is not None:
            ds = ds.take(max_len)
        return ds


class ReducerTestCase(unittest.TestCase):
    def test_reducer(self):

        max_len = None   # set to 100 to make it work

        ds = FramesDS().example_frames_ds(max_len)

        def key_fn(example_id, frame):
            return example_id

        def init_fn(example_id):
            return example_id, tf.zeros([0,], dtype=tf.int32)

        def reduce_fn(state, rinput):
            state_eid, frames = state
            example_id, frame = rinput
            tf.assert_equal(state_eid, example_id)
            frames = tf.concat([tf.reshape(frames, (tf.shape(frames)[0],
                                                    tf.shape(frame)[-1])),
                                tf.expand_dims(frame, axis=0)], axis=0)
            return example_id, frames

        def fin_fn(example_id, frames):
            return example_id, frames

        reducer = tf.data.experimental.Reducer(init_func=init_fn,
                                               reduce_func=reduce_fn,
                                               finalize_func=fin_fn)

        ds = ds.unbatch().batch(8)
        ds = ds.unbatch()

        def window_reduce_fn(key, ds):
            ds = ds.apply(tf.data.experimental.group_by_reducer(key_func=key_fn, reducer=reducer))
            return ds

        ds = ds.apply(tf.data.experimental.group_by_window(key_func=key_fn,
                                                           reduce_func=window_reduce_fn,
                                                           window_size=20))

        for example_id, frames in tqdm(ds):
            print("{}: {}".format(example_id, frames.shape))