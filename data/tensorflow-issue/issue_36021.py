import tensorflow as tf


class TestModule(tf.Module):
    def __init__(self, value):
        self.variable = tf.Variable(value)


module_1 = TestModule(value=9000)
tf.saved_model.save(module_1, "./foo")
module_2 = tf.saved_model.load("./foo")

assert module_1.variable.numpy() == module_2.variable.numpy()
assert module_1.trainable_variables == (module_1.variable, )
assert module_2.trainable_variables == (module_2.variable, )
assert module_1.trainable_variables == module_2.trainable_variables