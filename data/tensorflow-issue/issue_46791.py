import tensorflow as tf

class AutoregressiveGRU(layers.Layer):

    def __init__(
            self,
            output_dim: int,
            output_len: int,
            recurrent: layers.Recurrent,
            **kwargs,
    ):
        self.output_dim = output_dim
        self.output_len = output_len
        self.initial_state = None
        self.recurrent = recurrent
        super(AutoregressiveGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AutoregressiveGRU, self).build(input_shape)

    def call(self, x):
        outputs = []
        current_output = backend.zeros_like(backend.repeat(x, 1))
        current_state = x
        for _ in range(self.output_len):
            current_output, current_state = self.recurrent(
                current_output,
                initial_state=current_state,
            )
            outputs.append(current_output)
        result = layers.concatenate(outputs, axis=1)
        result = backend.reshape(result, (-1, self.output_len, self.output_dim))
        return result

def call(self, x):
        outputs = []
        current_output = backend.zeros_like(backend.repeat(x, 1))
        current_state = x
        i = tf.constant(0)
        c = lambda i, a, b, ta: i < 25
        def turn(i, current_statex, current_outputx, ta):
            current_output_cur, current_state_cur = self.recurrent(
                current_outputx,
                initial_state=current_statex,
            )
            ta.write(i, current_output_cur)
            tf.add(i, 1)
            return [i, current_state_cur, current_output_cur, ta]

        i, a, b, ta =  tf.while_loop(c, turn, [i, current_state, current_output, tf.TensorArray(tf.float32, size=25)], maximum_iterations=tf.constant(25))
        result = ta.stack()
        return backend.reshape(result, (-1, self.output_len, self.output_dim))