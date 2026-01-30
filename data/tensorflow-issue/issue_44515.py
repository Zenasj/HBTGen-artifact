import attr
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

@attr.s
class WrappedTensor(object):
    tensor = attr.ib()

@tf.function
def func_dict(x):
    def func_dict_per_replica(x):
        return x
    print('Result of Strategy.run(func_dict_per_replica):',
          tf.distribute.get_strategy().run(func_dict_per_replica, (x,)))

@tf.function
def func_attr(x):
    def func_attr_per_replica(x):
        return x
    print('Result of Strategy.run(func_attr_per_replica):',
          tf.distribute.get_strategy().run(func_attr_per_replica, (x,)))

with tf.distribute.MirroredStrategy().scope():
    data = tf.data.Dataset.from_tensors(tf.constant(0, shape=[2]))
    data = tf.distribute.get_strategy().experimental_distribute_dataset(data)
    v = next(iter(data))
    print()
    x = {'tensor': v}
    print('Input signature of func_dict:', func_dict.get_concrete_function(x).structured_input_signature)
    print()
    x = WrappedTensor(v)
    print('Input signature of func_attr:', func_attr.get_concrete_function(x).structured_input_signature)