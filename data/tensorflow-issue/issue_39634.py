class DropoutControl(Layer):

    def __init__(self, rate, seed=None, **kwargs):
        super(DropoutControl, self).__init__(**kwargs)
        self.rate = rate
        self.seed = seed
        self.cache_dropout_mask = None

    def reset_dropout(self):
        rate = ops.convert_to_tensor(
              self.rate, dtype=self.input_dtype, name="rate")
        random_tensor = random_ops.random_uniform(
            shape=self.shape, seed=self.seed, dtype=self.input_dtype)
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        keep_mask = random_tensor >= rate
        self.cache_dropout_mask  = scale * math_ops.cast(keep_mask, self.input_dtype)

    def get_dropout_mask(self):
        return self.cache_dropout_mask 

    def call(self, inputs, training):
        if self.cache_dropout_mask is None:
            self.shape = array_ops.shape(inputs)
            self.input_dtype = inputs.dtype
            self.reset_dropout()

        def dropped_inputs():
          return inputs * self.cache_dropout_mask

        output = tf_utils.smart_cond(training,
                                     dropped_inputs,
                                     lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'seed': self.seed
        }
        base_config = super(DropoutControl, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))