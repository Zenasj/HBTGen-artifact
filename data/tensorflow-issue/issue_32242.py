import tensorflow as tf

class A(tf.Module):
    def __init__(self):
        super().__init__()
        self.scalar = 1.
        self.variable = tf.Variable(1.)
        self.tensor = tf.convert_to_tensor(1.)
        self.list = [tf.convert_to_tensor(10.0), 20.0, tf.Variable(0.0)]

    def __str__(self):
        return f"scalar={self.scalar}, variable={self.variable.numpy()}, tensor={self.tensor}, list={self.list}"


a_module = A()
checkpoint = tf.train.Checkpoint(a=a_module)
manager = tf.train.CheckpointManager(checkpoint, '/tmp/example', max_to_keep=3)
manager.save()

print(f"1. {a_module}")

#### modify

a_module.scalar = -100.
a_module.variable.assign(123.)
a_module.tensor = tf.convert_to_tensor(-12.)
a_module.list = [3., 3.]

print(f"2. {a_module}")

#### restore
checkpoint.restore(manager.latest_checkpoint)

print(f"3. {a_module}")