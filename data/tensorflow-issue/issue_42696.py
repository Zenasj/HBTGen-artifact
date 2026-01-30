import random
from tensorflow import keras

python
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine import data_adapter
def test_weights(model, n_samples, use_weights=True):
    def test_data():
        def test_data_gen():
            n_classes = 5
            x = np.random.randn(n_samples,3)
            y = np.random.randint(0,n_classes,n_samples)
            yield (x.astype(np.float32),
                   y.astype(np.int32))
        gen_func = test_data_gen
        gen_types = (tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None])
        return gen_func, gen_types, gen_shapes
    gen_fn, gen_tp, gen_sh = test_data()
    ds_tst = tf.data.Dataset.from_generator(gen_fn, gen_tp, gen_sh)
    ds_tst = ds_tst.batch(2)
    ds_tst = ds_tst.prefetch(2)
    cw = {0 : 0.0, 1 : 1.5, 2 : 0.5,
          3 : 0.5, 4 : 0.5}
    data_handler = data_adapter.DataHandler(
        x=ds_tst,
        y=None,
        sample_weight=None,
        batch_size=None,
        steps_per_epoch=10,
        initial_epoch=0,
        epochs=1,
        shuffle=True,
        class_weight=(cw if use_weights else None),
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        model=model)
    print ('NEXT',next(iter(data_handler._dataset)))

model = tf.keras.Model()
test_weights(model, 5) # Always succeeds
test_weights(model, 50000) # Sometimes fails
test_weights(model, 50000, use_weights=False) # Always succeeds