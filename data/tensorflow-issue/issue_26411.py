import tensorflow as tf


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self, func=None):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = func

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


if __name__ == '__main__':

    def apply_kernel(tensor, kernel=tf.random_normal([3, 3])):
        t = tf.expand_dims(tensor, 0)
        t = tf.expand_dims(t, -1)
        k = tf.expand_dims(kernel, -1)
        k = tf.expand_dims(k, -1)

        # TODO the following line fails during Session.run
        #  call of tf.data.Iterator.get_next() (last line in this file)
        tf_conv = tf.nn.conv2d(t, k, [1, 1, 1, 1], "SAME")
        return tf.squeeze(tf_conv)

    def do_some_things(x, y):
        x = apply_kernel(x)
        return x, y

    n, image_shape = 100, [256, 256]

    ds = tf.data.Dataset.from_tensor_slices((
        tf.random_uniform([n] + image_shape), tf.random_uniform([n])
    ))
    ds = ds.map(do_some_things)
    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    data = iterator.get_next()
    ds_init_op = iterator.make_initializer(ds)

    with tf.train.SingularMonitoredSession(
            hooks=[IteratorInitializerHook(lambda s: s.run(ds_init_op))],
            config=tf.ConfigProto(log_device_placement=True)
    ) as sess:
        _ = sess.run(data)