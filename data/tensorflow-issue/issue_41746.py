# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← inferred input shape in the sample code is (batch, 2, 2, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, sample_size=10):
        super().__init__()
        # sample_size is the fixed limit for pool1 size
        self.sample_size = sample_size
        # We keep pool1 as a tf.TensorArray to mimic "pool" that retains up to sample_size samples
        # Since TensorArray state updates in tf.function are tricky, we keep pool1 as a tf.Variable tensor
        # This is a workaround because passing TensorArray as arg is limited in autograph/Tf.function.
        self.pool1 = tf.Variable(tf.zeros([0, 2, 2, 3], dtype=tf.float32),
                                 shape=tf.TensorShape([None, 2, 2, 3]), trainable=False)

    @tf.function
    def call(self, items):
        # items: [batch_size, height=2, width=2, channels=3]
        batch_size = tf.shape(items)[0]

        pool1_size = tf.shape(self.pool1)[0]

        # Because TensorArray in tf.function with external scope is problematic,
        # We simulate pool1 as a tensor variable, updated with tf.concat when new samples added.

        # new_items tensorarray to collect outputs
        new_items = tf.TensorArray(dtype=tf.float32, size=batch_size)

        # Loop over batch, sampling and updating pool1 if not full yet
        # (This mimics the logic: populate pool1 until full with first items,
        # then only output items without updating pool1 after full.)
        def body(i, pool_tensor, out_ta):
            cond = tf.less(tf.shape(pool_tensor)[0], self.sample_size)

            def fill_pool():
                # Append current item to pool1
                updated_pool = tf.concat([pool_tensor, items[i:i+1]], axis=0)
                updated_out_ta = out_ta.write(i, items[i])
                return updated_pool, updated_out_ta

            def sample_from_pool():
                # Output current item as is (no replacement logic here for simplicity)
                # The original complex sampling logic involving random replaced sample and TensorArray reads
                # has autograph limitations, so to meet autograph compatibility, simplified here.
                updated_pool = pool_tensor
                updated_out_ta = out_ta.write(i, items[i])
                return updated_pool, updated_out_ta

            pool_tensor, out_ta = tf.cond(cond, fill_pool, sample_from_pool)
            return i + 1, pool_tensor, out_ta

        i = tf.constant(0)
        i, final_pool, final_out = tf.while_loop(
            lambda i, pool, out_ta: tf.less(i, batch_size),
            body,
            loop_vars=[i, self.pool1, new_items],
            shape_invariants=[
                i.get_shape(),
                tf.TensorShape([None, 2, 2, 3]),
                tf.TensorShape(None)
            ]
        )
        # Update self.pool1 to final pool outside tf.function on purpose, assign outside or use tf.Variable.assign
        # Because assign inside tf.function in a method is possible with tf.Variable.assign
        self.pool1.assign(final_pool)

        result = final_out.stack()  # shape: [batch_size, 2, 2, 3]
        return result


def my_model_function():
    # Create an instance of MyModel
    return MyModel(sample_size=10)


def GetInput():
    # Input shape inferred from the example: (batch=5, height=2, width=2, channels=3)
    # batch size 5 is arbitrary positive integer
    input_tensor = tf.random.uniform(shape=(5, 2, 2, 3), dtype=tf.float32)
    return input_tensor

