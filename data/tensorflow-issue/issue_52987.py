import tensorflow as tf

experimental_implements = " ".join(['name: "addons:MaxUnpooling2D"'])
@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)], experimental_implements = experimental_implements)
def test(a, b):
    return a+b



class Test(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)], experimental_implements = experimental_implements)
    def test(self, a, b):
        return a+b

model = Test()
model2 = Test()
model.test = test
assert model.test._implements == test._implements
assert model2.test._implements == Test.test._implements