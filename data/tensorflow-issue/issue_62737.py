# tf.random.uniform((B, 32, 32, 4), dtype=tf.float32) ← Input tensor shape inferred from keras.Input(shape=(32,32,4))

import tensorflow as tf


class ImplicitA(tf.keras.layers.Layer):
    def __init__(self, mean, std, name, **kwargs):
        super(ImplicitA, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.impa = self.add_weight(
            name=self.name,
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.RandomNormal(
                mean=self.mean, stddev=self.std
            ),
            trainable=True
        )

    def call(self, x):
        # Cast x to impa dtype to avoid dtype mismatch issues
        return tf.cast(x, self.impa.dtype) + self.impa

    def get_config(self):
        config = super(ImplicitA, self).get_config()
        config.update({'mean': self.mean, 'std': self.std})
        return config


class ImplicitM(tf.keras.layers.Layer):
    def __init__(self, mean, std, name, **kwargs):
        super(ImplicitM, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.impm = self.add_weight(
            name=self.name,
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.RandomNormal(
                mean=self.mean, stddev=self.std
            ),
            trainable=True
        )

    def call(self, x):
        return tf.cast(x, self.impm.dtype) * self.impm

    def get_config(self):
        config = super(ImplicitM, self).get_config()
        config.update({'mean': self.mean, 'std': self.std})
        return config


class Model1(tf.keras.Model):
    # Implements model1 as described: ImplicitA -> Conv2D -> ImplicitM
    def __init__(self):
        super(Model1, self).__init__()
        self.impa = ImplicitA(mean=0.0, std=0.02, name='impa')
        self.conv = tf.keras.layers.Conv2D(filters=8, kernel_size=1, strides=1, name='conv')
        self.impm = ImplicitM(mean=0.0, std=0.02, name='impm')

    def call(self, x):
        x = self.impa(x)
        x = self.conv(x)
        x = self.impm(x)
        return x


class Model2(tf.keras.Model):
    # Implements model2 as described: ImplicitA -> Conv2D (no ImplicitM)
    def __init__(self):
        super(Model2, self).__init__()
        self.impa = ImplicitA(mean=0.0, std=0.02, name='impa')
        self.conv = tf.keras.layers.Conv2D(filters=8, kernel_size=1, strides=1, name='conv')

    def call(self, x):
        x = self.impa(x)
        x = self.conv(x)
        return x


class Model3(tf.keras.Model):
    # Implements model3 as described: ImplicitA -> ImplicitM -> Conv2D
    def __init__(self):
        super(Model3, self).__init__()
        self.impa = ImplicitA(mean=0.0, std=0.02, name='impa')
        self.impm = ImplicitM(mean=0.0, std=0.02, name='impm')
        self.conv = tf.keras.layers.Conv2D(filters=8, kernel_size=1, strides=1, name='conv')

    def call(self, x):
        x = self.impa(x)
        x = self.impm(x)
        x = self.conv(x)
        return x


class MyModel(tf.keras.Model):
    """
    Fused model encapsulating Model1, Model2 and Model3 as submodules.
    The forward call runs all three models on the input, and returns a dictionary
    of outputs as well as a boolean tensor indicating whether Model1 output differs
    from Model2 and Model3 outputs.

    This captures the main models discussed and the issue context,
    allowing comparative use.

    Output dictionary keys:
        'model1': output tensor of Model1
        'model2': output tensor of Model2
        'model3': output tensor of Model3
        'model1_eq_others': boolean tensor indicating if model1 output equals
                           model2 and model3 outputs within a tolerance.
    """

    def __init__(self, rtol=1e-5, atol=1e-8):
        super(MyModel, self).__init__()
        self.model1 = Model1()
        self.model2 = Model2()
        self.model3 = Model3()
        self.rtol = rtol
        self.atol = atol

    def call(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        # Check if model1 output matches model2 and model3 outputs elementwise within tolerance
        eq_1_2 = tf.reduce_all(tf.math.abs(out1 - out2) <= self.atol + self.rtol * tf.math.abs(out2))
        eq_1_3 = tf.reduce_all(tf.math.abs(out1 - out3) <= self.atol + self.rtol * tf.math.abs(out3))
        model1_eq_others = tf.logical_and(eq_1_2, eq_1_3)

        return {
            'model1': out1,
            'model2': out2,
            'model3': out3,
            'model1_eq_others': model1_eq_others
        }


def my_model_function():
    # Return an instance of MyModel with default tolerances
    return MyModel()


def GetInput():
    # Return a random float32 tensor matching input shape (batch=1, H=32, W=32, C=4)
    # This matches the keras.Input(shape=(32,32,4)) used in original models.
    return tf.random.uniform((1, 32, 32, 4), dtype=tf.float32)

