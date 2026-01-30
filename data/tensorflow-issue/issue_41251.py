from tensorflow import keras

import tensorflow as tf

class Boo():
    def __init__(self, booarg1, booarg2):
        print("Boo Class: ", booarg1)
        print("Boo Class: ", booarg2)

class FooModel(tf.keras.Model):
    def __init__(self, fooarg, **kwargs):
        super(FooModel, self).__init__(**kwargs)
        print("Foo Class: ", fooarg)
        boo = Boo(**kwargs)

foo = FooModel(fooarg=1, booarg1=2, booarg2=3)