import random
import tensorflow as tf

@tf.function
def f(X):
    tf.random.set_seed(X)

f(tf.constant(1))

@tf.function
def f(X):
    tf.print(tf.random.normal((2,3)))

f(tf.constant(1))

@tf.function
def f(X):
    tf.random.set_seed(X)
    tf.print(tf.random.normal((2,3)))

f(tf.constant(1))

@tf.function
def f():
    tf.random.set_seed(1)
    tf.print(tf.random.normal((2,3)))

f()
f()

@tf.function
def f(X):
    tf.print(tf.random.normal((2,3), seed=X))

f(tf.constant(1))

class MyModule(tf.Module):
    def __init__(self, seed):
        self.seed = seed
        self.strategy = tf.distribute.MirroredStrategy()

    def _train(self):
        # this works but all replicas have the same seed.
        tf.print(tf.random.normal((2,3), seed=self.seed))

        if False: # replace with True to test
            # the seed differs but this fails with OperatorNotAllowedInGraphError
            repl_id = tf.distribute.get_replica_context().replica_id_in_sync_group
            print(tf.random.normal((2,3), seed=self.seed + repl_id))
    
    @tf.function
    def distributed_train(self):
        self.strategy.run(self._train)

    
module = MyModule(seed=1)
module.distributed_train()

@tf.function
def f():
    tf.random.set_seed(1)
    tf.print(tf.random.normal((2,3)))

f()

@tf.function
def f(X):
    tf.random.set_seed(X)
    tf.print(tf.random.normal((2,3)))

f(tf.constant(1))

@tf.function
def f(X):
    if X == (0,0):
      return True
    return False

print(f(1))
print(f(tf.constant(1))) # This will not work

@tf.function
def f():
    tf.random.set_seed(1)
    tf.print(tf.random.stateless_normal([3], seed=[2,4]))

f()
f()

tf.print((self.seed + repl_id).dtype)

class MyModule(tf.Module):
    def __init__(self, seed):
        self.seed = seed
        self.strategy = tf.distribute.MirroredStrategy()

    def _train(self):
        repl_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        tf.print('replica', repl_id, tf.random.stateless_normal((3,), seed=[self.seed + repl_id, self.seed + repl_id]))
    
    @tf.function
    def distributed_train(self):
        self.strategy.run(self._train)

    
module = MyModule(seed=0)
module.distributed_train()
module.distributed_train()

module = MyModule(seed=3)
module.distributed_train()
module.distributed_train()