from tensorflow.keras import layers

class InstanceNormalization(keras.layers.Layer):

    def __init__(self, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.epsilon=epsilon

    def build(self, batch_input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=batch_input_shape[-1:],
            initializer=tensorflow.random_normal_initializer(1,0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=batch_input_shape[-1:],
            initializer="zeros",
            trainable=True
        )
        self.axis = range(1, len(batch_input_shape)-1)
        super().build(batch_input_shape)

    def call(self, x):
        mean = keras.backend.mean(x, axis=self.axis, keepdims=True)
        variance = keras.backend.mean(keras.backend.square(x-mean), axis=self.axis, keepdims=True)
        normalized = (x - mean) / keras.backend.sqrt(variance+self.epsilon)
        return self.scale * normalized + self.offset