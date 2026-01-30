import tensorflow as tf

print(tf.__version__)

class MyObj:
    def __init__(self):
        self.value = 0

obj = MyObj()

@tf.function
def with_py_side_effect(tensorflow_stuff, o):

    # do my complex tf stuff
    ...

    o.value += 1
    return tensorflow_stuff

for i in range(5):
    print(i, obj.value)
    a = with_py_side_effect(None, obj)