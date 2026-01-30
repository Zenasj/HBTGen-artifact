from tensorflow.keras import layers

class ParametricScalar(keras.layers.Layer):
    """combine multiple activations weighted by learnable variables"""
    def __init__(self, alpha_initializer='ones', shared_axes=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_initializer = keras.initializers.get(alpha_initializer)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def get_config(self):
        return {'act_set': self.act_set}

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.alpha = self.add_weight(
            shape=param_shape,
            name='alpha',
            initializer=self.alpha_initializer)
        
    def get_config(self):
        config = {
            'alpha_initializer': keras.initializers.serialize(self.alpha_initializer),
            'shared_axes': self.shared_axes
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        return inputs * self.alpha