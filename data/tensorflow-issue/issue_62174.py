import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


# Let's create a model first
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self._l1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3
        )
        self._l1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3
                ),
                tf.keras.layers.GlobalAveragePooling2D(keepdims=False),
                tf.keras.layers.Dense(units=10)
            ]
        )
        self._l2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3
                ),
                tf.keras.layers.GlobalAveragePooling2D(keepdims=False),
                tf.keras.layers.Dense(units=10)
            ]
        )
        self._l3 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3
                ),
                tf.keras.layers.GlobalAveragePooling2D(keepdims=False),
                tf.keras.layers.Dense(units=10)
            ]
        )

    def call(self, inputs, training=None, mask=None):
        y1 = self._l1(inputs)
        y2 = self._l2(inputs)
        y3 = self._l3(inputs)
        out = tf.stack([y1, y2, y3])
        # Output shape is (3, batch_size, 10)

        return out


model = MyModel()


# just a random image classification dataset
def get_dataset():
    db = tf.data.Dataset.range(100)
    db = db.map(
        lambda x: (tf.random.uniform(shape=(224, 224, 3), dtype=tf.float32),
                   tf.one_hot(tf.random.uniform(shape=(), minval=0, maxval=9, dtype=tf.int64), depth=10)
                   )
    )
    db = db.batch(5)
    return db


dataset = get_dataset()


# Computes the loss. But gradient backpropagation fails.
# Here tf.map_fn is used to compute the loss of the output of l1, l2 and l3 ( See Model ) with respect to GT labeles.
def get_loss_v1(logits, labels):
    loss = tf.map_fn(
        fn=lambda x: tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                from_logits=True,
                y_pred=tf.gather(logits, x),
                y_true=labels
            )
        ),
        elems=tf.range(3),
        fn_output_signature=tf.float32
    )
    return loss


# Computes the loss. But gradient backpropagation fails.
# Here appending to a list is used to compute the loss of the output of l1, l2 and l3 ( See Model ) with respect to GT labeles.
# This method should be avoided as it makes use of python list. TensorArray is recommended to be used here.

# Check out https://github.com/tensorflow/tensorflow/issues/37512
def get_loss_v2(logits, labels):
    losses = list()
    for i in tf.range(3):
        y_pred = tf.gather(logits, i)
        losses.append(
            tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    from_logits=True,
                    y_pred=y_pred,
                    y_true=labels
                )
            )
        )
    return losses


# Computes the loss. But gradient backpropagation fails.
# Here tf.TensorArray with tf.while_loop is used to compute the loss of the output of l1, l2 and l3 ( See Model ) with respect to GT labeles.
def get_loss_v3(logits, labels):
    losses = tf.TensorArray(dtype=tf.float32,
                            size=0, dynamic_size=True, clear_after_read=False, element_shape=()
                            )
    index = tf.constant(0)

    def body(counter_var, log, lbl, l):
        l = l.write(l.size(), tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                from_logits=True,
                y_pred=tf.gather(log, counter_var),
                y_true=lbl
            )
        ))
        return counter_var + 1, log, lbl, l

    output = tf.while_loop(
        cond=lambda i, *_: tf.less(i, 3),
        loop_vars=(index, logits, labels, losses),
        body=body,
        parallel_iterations=1
    )
    loss_val = output[-1].stack()
    return loss_val


def train_step(images, labels):
    with tf.GradientTape(persistent=True) as tape:
        logits = model(images, training=True)
        loss_val = get_loss_v1(logits, labels)
        # loss_val = get_loss_v2(logits, labels)
        # loss_val = get_loss_v3(logits, labels)
        print(loss_val)  # This prints the forward losses properly
    grads = list()
    for i in tf.range(3):
        tgt = tf.gather(loss_val, i)
        grads.append(
            tape.gradient(
                target=tgt,
                sources=model.trainable_variables
            )
        )
    print(grads)  # This always prints None


for data in dataset:
    images, labels = data
    train_step(images, labels)