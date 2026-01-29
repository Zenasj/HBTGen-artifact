# tf.random.uniform((batch_size, 224, 224, 3), dtype=tf.float32) ← input shape inferred from dataset generator in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define three parallel subnetworks l1, l2, l3
        # Each has Conv2D -> GlobalAveragePooling2D -> Dense(10)
        self._l1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation=None),
                tf.keras.layers.GlobalAveragePooling2D(keepdims=False),
                tf.keras.layers.Dense(units=10)
            ]
        )
        self._l2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation=None),
                tf.keras.layers.GlobalAveragePooling2D(keepdims=False),
                tf.keras.layers.Dense(units=10)
            ]
        )
        self._l3 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation=None),
                tf.keras.layers.GlobalAveragePooling2D(keepdims=False),
                tf.keras.layers.Dense(units=10)
            ]
        )

    def call(self, inputs, training=None, mask=None):
        y1 = self._l1(inputs)  # shape: (batch_size, 10)
        y2 = self._l2(inputs)  # shape: (batch_size, 10)
        y3 = self._l3(inputs)  # shape: (batch_size, 10)
        # Stack along 0th dim to get shape (3, batch_size, 10)
        out = tf.stack([y1, y2, y3], axis=0)
        return out


def fused_loss_vmap(logits, labels):
    """
    Compute categorical cross-entropy loss per model output (3 outputs),
    with gradients supported in eager and graph modes.

    This function uses tf.vectorized_map (which supports backpropagation better than tf.map_fn).
    """
    def loss_fn(idx):
        pred = tf.gather(logits, idx)  # shape (batch_size, 10)
        # Compute mean CCE for this output
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true=labels,
                y_pred=pred,
                from_logits=True
            )
        )
        return loss
    losses = tf.vectorized_map(loss_fn, tf.range(3))
    return losses


def fused_loss_while_loop(logits, labels):
    """
    Compute categorical cross-entropy losses over the 3 model outputs using tf.while_loop and TensorArray.
    This approach was known from the issue to not propagate gradients properly,
    but included for completeness and as reference.
    """
    losses_ta = tf.TensorArray(dtype=tf.float32, size=3, dynamic_size=False, clear_after_read=False)
    i = tf.constant(0)

    def cond(i, ta):
        return tf.less(i, 3)

    def body(i, ta):
        pred = tf.gather(logits, i)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true=labels,
                y_pred=pred,
                from_logits=True
            )
        )
        ta = ta.write(i, loss)
        return i+1, ta

    i, losses_ta = tf.while_loop(cond, body, loop_vars=[i, losses_ta], parallel_iterations=1)
    return losses_ta.stack()


def fused_loss_python_loop(logits, labels):
    """
    Compute categorical cross-entropy losses using a python for-loop and list.
    This works in eager but is not recommended as it breaks graph mode optimizations.
    """
    losses = []
    for i in range(3):
        pred = tf.gather(logits, i)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true=labels,
                y_pred=pred,
                from_logits=True
            )
        )
        losses.append(loss)
    return tf.stack(losses)


class MyModelWithLoss(tf.keras.Model):
    """
    A wrapper model to run forward pass and compute losses fused from submodels (l1,l2,l3)
    and to provide a comparison of losses from different implementations,
    highlighting the gradients issue in TensorArray / tf.map_fn approaches.

    The forward pass outputs logits stacked (3, batch_size, 10).

    The call returns a dict of losses computed with different methods and a boolean tensor
    indicating if the three loss computations agree (within tolerance).
    """
    def __init__(self):
        super(MyModelWithLoss, self).__init__()
        self.base_model = MyModel()

    def call(self, inputs, labels, training=None):
        logits = self.base_model(inputs, training=training)
        loss_vmap = fused_loss_vmap(logits, labels)
        loss_while_loop = fused_loss_while_loop(logits, labels)
        loss_py_loop = fused_loss_python_loop(logits, labels)

        # Compare all losses with small tolerance
        # Because Python loop outputs and vmap should be same,
        # While_loop may differ / gradient may fail
        eps = 1e-6
        vmap_vs_py = tf.less_equal(tf.abs(loss_vmap - loss_py_loop), eps)
        vmap_vs_while = tf.less_equal(tf.abs(loss_vmap - loss_while_loop), eps)
        all_equal = tf.reduce_all(tf.logical_and(vmap_vs_py, vmap_vs_while))

        return {
            'loss_vmap': loss_vmap,
            'loss_while_loop': loss_while_loop,
            'loss_python_loop': loss_py_loop,
            'losses_all_close': all_equal
        }


def MyModelFactory():
    # Return the wrapped model that provides fused loss computations for the 3 networks
    return MyModelWithLoss()


def GetInput():
    # Generate a batch of random images and random one-hot labels consistent with example in issue
    batch_size = 5  # From original dataset batch size
    img_shape = (224, 224, 3)
    num_classes = 10

    images = tf.random.uniform(shape=(batch_size,) + img_shape, dtype=tf.float32)
    labels_int = tf.random.uniform(shape=(batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
    labels = tf.one_hot(labels_int, depth=num_classes)

    return (images, labels)

