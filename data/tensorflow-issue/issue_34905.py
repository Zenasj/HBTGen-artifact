import numpy as np
import tensorflow as tf

class A:
    @tf.function(experimental_relax_shapes=True)
    def f(self, data):
        return tf.reduce_sum(data)

@tf.function(experimental_relax_shapes=True)
def f(data):
    return tf.reduce_sum(data)

a = A()
for i in range(100):
    print(a.f(np.ones(i)))

print(f(np.ones(i)))