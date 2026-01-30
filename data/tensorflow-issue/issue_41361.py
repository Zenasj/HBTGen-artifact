from tensorflow import keras

#%pip install tensorflow==2.2.0
#%pip install tensorflow==2.1.1
import tensorflow as tf

inp = tf.keras.Input(3)
model = tf.keras.Model(inp, inp)

model.compile(loss="bce", metrics=["acc"])
model.evaluate(tf.constant([[0.1, 0.6, 0.9]]), tf.constant([[0, 1, 1]]))

if tf.version.VERSION == "2.2.0":
    print(model.compiled_metrics.metrics[0]._fn.__name__)
else:
    print(model._per_output_metrics[0]["acc"]._fn.__name__)