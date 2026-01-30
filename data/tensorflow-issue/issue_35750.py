import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

input = tf.keras.Input(...)
y = tf.keras.layers.Dense(...)(input)
...
logits = tf.keras.layers.Dense(..., name="logit_layer")(y)

model = tf.keras.models.Model(inputs, logits)

def my_loss(labels, logits):
    # here just wrapping a known loss to remove errors that could come from me
    # however there may be additional functionality here
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )
    return loss

model.compile(
    ...,
    loss ={
        'logit_layer': my_loss
    },
)

model.save(...)
tf.keras.experimental.export_saved_model(model, ...)
tf.keras.models.save_model(model, ...)
model.save_weights(...)

lambda y_pred, y_true: y_pred