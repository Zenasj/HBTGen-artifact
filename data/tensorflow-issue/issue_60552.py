# tf.random.normal((batch_size, input_features), dtype=tf.float32) ← Input will be a 2D float tensor, assuming input features=3 here

import tensorflow as tf

class FlexibleDenseModule(tf.keras.layers.Layer):
    # Adapted from tf.Module example to tf.keras.layers.Layer for better TF2 compatibility and easier usage
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.is_built = False

    def build(self, input_shape):
        # Create variables once input shape is known (on build)
        in_features = input_shape[-1]
        self.w = self.add_weight(
            shape=(in_features, self.out_features),
            initializer='random_normal',
            trainable=True,
            name='w'
        )
        self.b = self.add_weight(
            shape=(self.out_features,),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        self.is_built = True
        super().build(input_shape)

    def call(self, inputs):
        # If not built, build now with shape inference (some keras functions can do this)
        if not self.is_built:
            self.build(inputs.shape)
        y = tf.matmul(inputs, self.w) + self.b
        return tf.nn.relu(y)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Composed of two FlexibleDenseModules like the original composite module 
        self.dense_1 = FlexibleDenseModule(out_features=3)
        self.dense_2 = FlexibleDenseModule(out_features=2)

    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x


def my_model_function():
    # Instantiate and return MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input compatible with FlexibleDenseModule expecting input feature dimension = 3
    # Batch size = 1 for simplicity (can be any positive integer)
    return tf.random.uniform(shape=(1, 3), dtype=tf.float32)

