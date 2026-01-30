import tensorflow as tf

class A:
    def foo(self, x):
        return x + 1

class B(A):
    @tf.function
    def bar(self, x):
        return super().foo(x)

b = B()
b.bar(5) # raises RuntimeException

import tensorflow as tf

class A:
    def foo(self, x):
        return x + 1

class B(A):
    @tf.function(autograph=False)
    def bar(self, x):
        return super().foo(x)

b = B()
b.bar(5) # okay, returns 6

import tensorflow as tf

class A:
    def foo(self, x):
        return x + 1

class B(A):
    def __init__(self):
        self._super = super()
    @tf.function
    def bar(self, x):
        return self._super.foo(x)

b = B()
b.bar(5) # okay, returns 6