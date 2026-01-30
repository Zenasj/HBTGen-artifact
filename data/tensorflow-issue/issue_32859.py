import tensorflow as tf


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        with tf.variable_scope("layer_norm"):
            self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size], 
              initializer=tf.ones_initializer())
            self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size], 
              initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


if __name__ == "__main__":
    x = tf.random_uniform((23, 29))
    ln = LayerNormalization(29)

    y = ln(x)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(y)