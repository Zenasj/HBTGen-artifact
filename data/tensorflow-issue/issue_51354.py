import random

import tensorflow as tf

from tensorflow.keras.losses import Loss

class EagerOom(Loss):

    def __init__(self):
        super(EagerOom, self).__init__()

    def __call__(self, targets, logits):
        logits = tf.reshape(logits, [-1])
        targets = tf.reshape(targets, [-1])
        return self.inner(targets, logits)


    @tf.custom_gradient
    def inner(self, targets, logits):

        g_logits = tf.zeros_like(logits)

        for idx in range(len(targets)):
            print(f"idx: {idx}")
            g_logits_upd = logits - logits[idx]
            g_logits += g_logits_upd
        
        loss = tf.reduce_mean(targets * logits)

        def grad_fn(upstream):
            g_targets = tf.zeros_like(targets)
            return g_targets, g_logits

        return loss, grad_fn



def main():
    tf.random.set_seed(0)
    # b, a, c = 1, 1000, 5 # small problem
    b, a, c = 1, 10000, 5 # large problem
    logits = tf.random.normal((b, a, c))
    targets = tf.random.uniform((b, a), 0, c + 1, dtype=tf.int32)
    targets = tf.one_hot(targets, depth=c, axis=-1)
    loss_fn = EagerOom()
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(logits)
        loss = loss_fn(targets, logits)
    print("Done.")


if __name__ == '__main__':
    main()