import tensorflow as tf

input = x
my_layer = MyLayer(...)
x = my_layer(x)
...
model = Model(input, x)
model.compile(...)
model.fit(..., callbacks=[WarmStartCallback(my_layer.alpha)])

class MyLayer(Layer):
    def __init__(self, warm_start=True):
        self.alpha = tf.Variable(...)
        if warm_start:
            self.add_callback(WarmStartCallback(self.alpha))


input = x
x = MyLayer(...)(x)
...
model = Model(input, x)
model.compile(...)
model.fit(...)