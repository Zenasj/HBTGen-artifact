import numpy as np

class MyDatabase:
    def __len__(self): return ...
    def __getitem__(self, item): return ...  # slow code without gitlock, i.e. io and numpy

def generator():
    db = MyDatabase()
    yield from [db[i] for i in range(len(db))]

import time

start = time.perf_counter()
    
def body(i):
    global start
    if i == 0:
        start = time.perf_counter()
    time.sleep(0.2)
    return np.array([float(i), time.perf_counter() - start])

def gen():
    for i in range(5):
        yield body(i)
        
ds = tf.data.Dataset.from_generator(gen, tf.float64)
ds = ds.prefetch(5)
iterator = ds.make_one_shot_iterator()

entry = iterator.get_next()

with tf.Session() as sess:
    
    try:
        while True:
            print(sess.run(entry))
    except tf.errors.OutOfRangeError:
        pass
# Serial execution:  [index, time from start of first load to return of current load]
# [ 0.          0.20034038]
# [ 1.          0.40189139]
# [ 2.          0.60322792]
# [ 3.          0.80472201]
# [ 4.          1.00612245]

ds = tf.data.Dataset.range(5)

def map_func(i):
    return tf.py_func(body, [i], tf.float64, stateful=False)

ds = ds.map(map_func, num_parallel_calls=4)
ds = ds.prefetch(1)
iterator = ds.make_one_shot_iterator()

entry = iterator.get_next()

with tf.Session() as sess:
    
    try:
        while True:
            print(sess.run(entry))
    except tf.errors.OutOfRangeError:
        pass

# Parallel execution:  [index, time from start of first load to return of current load]
# [ 0.          0.20026697]
# [ 1.          0.20084557]
# [ 2.          0.20095535]
# [ 3.          0.20048737]
# [ 4.          0.40154806]

import tensorflow as tf
import traceback

from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import script_ops

def my_py_func(func, args=(), kwargs={}, output_types=None, output_shapes=None, stateful=True, name=None):
    # Low level function
    
    if isinstance(args, list):
        # Force tuple, nest.flatten interprets list as scalar
        args = tuple(args)
        
    if callable(output_types):
        # If callable, assume same signature and call with tensors and get the types
        output_types = output_types(*args, **kwargs)
    if callable(output_shapes):
        # If callable, assume same signature and call with tensors and get the shapes
        output_shapes = output_shapes(*args, **kwargs)
    
    flat_output_types = nest.flatten(output_types)
    
    args = (args, kwargs)
        
    flat_args = nest.flatten(args)

    def python_function_wrapper(*py_args):
        try:
            py_args, py_kwargs = nest.pack_sequence_as(args, py_args)
            ret = func(*py_args, **py_kwargs)
            nest.assert_shallow_structure(output_types, ret)
        except Exception:
            traceback.print_exc()
            raise
        return nest.flatten(ret)
    
    flat_values = script_ops.py_func(
      python_function_wrapper, flat_args, flat_output_types, stateful=stateful, name=name)

    if output_shapes is not None:
        # I am not sure if this is nessesary
        output_shapes = nest.map_structure_up_to(
            output_types, tensor_shape.as_shape, output_shapes)
        flattened_shapes = nest.flatten(output_shapes)
        for ret_t, shape in zip(flat_values, flattened_shapes):
            ret_t.set_shape(shape)

    return nest.pack_sequence_as(output_types, flat_values)
    
def py_func_decorator(output_types=None, output_shapes=None, stateful=True, name=None):
    def decorator(func):
        def call(*args, **kwargs):
            return my_py_func(
                func, 
                args, kwargs, 
                output_types=output_types, output_shapes=output_shapes, 
                stateful=stateful, name=name
            )
        return call
    return decorator
            
@py_func_decorator(
    output_types=lambda a, b: {
        'a': nest.map_structure(lambda x: x.dtype, a), 
        'b': nest.map_structure(lambda x: x.dtype, b),
    }, 
    output_shapes=lambda a, b: {
        'a': nest.map_structure(lambda x: x.shape, a), 
        'b': nest.map_structure(lambda x: x.shape, b),
    },
)
def foo(a, b):
    return {'a': a, 'b': b}
    
def bar(a, b):
    return {'a': a, 'b': b}
    
a = tf.constant([4., 5], name='a')
b = tf.constant([4., 5], name='b')
c = tf.constant([4., 5], name='c')
out0 = my_py_func(
    bar, [a, b], output_types={'a': a.dtype, 'b': b.dtype}, output_shapes={'a': a.shape, 'b': b.shape})
out1 = foo(a, b)
out2 = foo(a, b=b)
out3 = foo(a=a, b=b)
out4 = foo(a=dict(a=a, c=c), b=b)
print('out0', out0)
print('out1', out1)

from IPython.lib.pretty import pprint

with tf.Session() as sess:
    pprint(sess.run([out0]))
    pprint(sess.run([out1, out2, out3]))
    pprint(sess.run([out3]))