import tensorflow as tf
class ModuleLevel2(tf.Module):
    
    def __init__(self):
        self.state = tf.TensorArray(size=0, dynamic_size=True, clear_after_read=False, dtype=tf.float32)

    def __call__(self, x):
        self.state = self.state.write(self.state.size(), x)
        x = self.state.stack()
        x = tf.identity(x) # Some op that requires all saved inputs
        tf.print(x)

class ModuleLevel1(tf.Module):

    def __init__(self):
        self.module2 = ModuleLevel2()

    def __call__(self, x):
        self.module2(x)

class Model(tf.Module):

    def __init__(self):
        self.module1 = ModuleLevel1()

    def __call__(self, x):
        self.module1(x)

model = Model()

@tf.function
def search():
    for i in range(10):
        x = (i*5)+tf.constant([1,2,3,4,5], dtype=tf.float32)
        model(x)

search()

import tensorflow as tf

class ModuleLevel2(tf.Module):
    
    def __init__(self):
        pass

    def __call__(self, x, state):
        state = state.write(state.size(), x)
        x = state.stack()
        x = tf.identity(x) # Some op that requires all saved inputs
        tf.print(x)
        return state

class ModuleLevel1(tf.Module):

    def __init__(self):
        self.module2 = ModuleLevel2()

    def __call__(self, x, state):
        state = self.module2(x, state)
        return state

class Model(tf.Module):

    def __init__(self):
        self.module1 = ModuleLevel1()

    def __call__(self, x, state):
        state = self.module1(x, state)
        return state

model = Model()

@tf.function
def search():
    state = tf.TensorArray(size=0, dynamic_size=True, clear_after_read=False, dtype=tf.float32)
    for i in range(10):
        x = (i*5)+tf.constant([1,2,3,4,5], dtype=tf.float32)
        state = model(x, state)

search()