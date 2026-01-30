import tensorflow as tf

class Base(tf.Module):
    def __call__(self, x):
        return x + 1.

class Sub(Base):
    def __call__(self, x):
        return super().__call__(x) if True else 1.

@tf.function
def test():
    return Sub()(tf.constant(42.))

print(test())