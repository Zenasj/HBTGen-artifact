import tensorflow as tf
print(tf.__version__)

class First:
    def __init__(self, initial: int):
        self.value = tf.constant(initial)

    def increment(self):
        self.value += 1

    def __str__(self):
        return f'Object with tensor = {self.value}'

@tf.function
def increment(obj: First):
    obj.increment()

c1 = First(100)

c1.increment()
print(c1)

increment(c1)
print(c1)