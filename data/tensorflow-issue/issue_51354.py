# tf.random.normal((b, a, c), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.loss_fn = EagerOom()

    def call(self, inputs):
        # inputs is expected to be a tuple of (targets, logits)
        targets, logits = inputs
        return self.loss_fn(targets, logits)


class EagerOom(tf.keras.losses.Loss):
    def __init__(self):
        super(EagerOom, self).__init__()

    def __call__(self, targets, logits):
        # Flatten the targets and logits to 1D for the loss computation as in original code
        logits = tf.reshape(logits, [-1])
        targets = tf.reshape(targets, [-1])
        return self.inner(targets, logits)

    @tf.custom_gradient
    def inner(self, targets, logits):
        # To avoid TensorFlow keeping references accumulating memory during the loop,
        # stop gradient recording on inputs, as suggested in the discussion.
        targets = tf.stop_gradient(targets)
        logits = tf.stop_gradient(logits)

        # Initialize a zero gradient tensor for logits
        g_logits = tf.zeros_like(logits)

        # Accumulate the sum of (logits - logits[idx]) over all idx,
        # replicating the original computation.
        # This loop is memory-heavy and causes OOM on large inputs.
        for idx in tf.range(tf.shape(targets)[0]):
            g_logits_upd = logits - logits[idx]
            g_logits += g_logits_upd

        loss = tf.reduce_mean(targets * logits)

        def grad_fn(upstream):
            # Gradient of loss wrt targets is zero tensor,
            # gradient wrt logits is accumulated g_logits nonetheless
            g_targets = tf.zeros_like(targets)
            # upstream is scalar gradient; multiply gradients by upstream scalar
            return g_targets * upstream, g_logits * upstream

        return loss, grad_fn


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Following original main() code:
    # b, a, c = 1, 10000, 5 # large problem, can reduce if memory is tight
    b, a, c = 1, 1000, 5  # Using smaller shape to avoid OOM in typical environments

    # logits: shape (b, a, c), float32 normal
    logits = tf.random.normal((b, a, c), dtype=tf.float32)
    # targets: shape (b, a), int32, uniform between 0 and c (inclusive)
    targets_raw = tf.random.uniform((b, a), 0, c + 1, dtype=tf.int32)
    # one-hot encode targets to shape (b, a, c)
    targets = tf.one_hot(targets_raw, depth=c, axis=-1, dtype=tf.float32)

    return (targets, logits)

