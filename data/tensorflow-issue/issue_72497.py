x = Input(...)
...
tf_fn(x)  # Invalid.

class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)