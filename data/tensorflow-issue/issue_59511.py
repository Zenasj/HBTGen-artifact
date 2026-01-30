import random
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
from tensorflow.experimental import numpy as tnp
import numpy as np

def set_tpu(mixed_precision=True):
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() 
        if mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_bfloat16") 
        tf.config.set_soft_device_placement(False)
        strategy = tf.distribute.TPUStrategy(tpu)
        physical_devices = tf.config.list_logical_devices('TPU')
        return (strategy, physical_devices)

    
mxp = False
jit = False
strategy, physical_devices = set_tpu(mixed_precision=mxp)
physical_devices, tf.__version__

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_x = tf.Variable((
            tnp.empty((0, 32), dtype=tf.float32)), shape=[None, 32]
        )
        self.val_gt = tf.Variable(
            tnp.empty((0), dtype=tf.float32), shape=[None]
        )
        self.val_pred = tf.Variable(
            tnp.empty((0, 1), dtype=tf.float32), shape=[None, 1]
        )
        
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        
        # ATTENTION 
        # Main Cause !!!
        self.val_x.assign(
            tf.concat([self.val_x, x], axis=0)
        )
        self.val_gt.assign(
            tf.concat([self.val_gt, y], axis=0)
        )
        self.val_pred.assign(
            tf.concat([self.val_pred, y_pred], axis=0)
        )
        return {m.name: m.result() for m in self.metrics}

with strategy.scope():
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1, dtype='float32')(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer="adam", loss="mse", metrics=["mae"],
    )

x = np.random.random((1000, 32))
y = np.random.random((1000))
x_test = np.random.random((10, 32))
y_test = np.random.random((10))

model.fit(
    x, 
    y, 
    epochs=5, 
    validation_data=(x_test, y_test),
    verbose=2, 
    batch_size=4,
)

class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xywh",
            # passing 1e9 ensures we never evaluate until
            # `metrics.result(force=True)` is
            # called.
            evaluate_freq=1e9,
        )

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in tqdm.tqdm(self.data):
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)
        return logs

model.fit(
    train_ds.take(20),
    validation_data=eval_ds.take(20),
    # Run for 10-35~ epochs to achieve good scores.
    epochs=1,
    callbacks=[EvaluateCOCOMetricsCallback(eval_ds.take(20))],
)