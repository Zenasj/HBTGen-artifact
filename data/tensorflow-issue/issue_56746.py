import tensorflow as tf

N_DATASETS_TO_INTERLEAVE = 10

@tf.autograph.experimental.do_not_convert
def hello(idx):
  for j in range(idx):
    yield f"IDX: {idx}"
    

@tf.autograph.experimental.do_not_convert
def interleave_fn(_):
    print("[INFO] Calling Interleave Fn")         # THIS LINE SHOULD BE PRINTED `N_DATASETS_TO_INTERLEAVE` times. Only appears once.
    return tf.data.Dataset.from_generator(
        hello, args=(_,), output_types=tf.string
    )


ds = tf.data.Dataset.range(N_DATASETS_TO_INTERLEAVE).interleave(
    interleave_fn
)

options = tf.data.Options()
options.experimental_optimization.apply_default_optimizations = False
ds = ds.with_options(options)


@tf.autograph.experimental.do_not_convert
def get_dataset(_ds):
    for x in iter(_ds):
        yield x


for x in get_dataset(ds):
  print(x)

import tensorflow as tf

from tensorflow.python.data.ops import structured_function

@tf.autograph.experimental.do_not_convert  # Not doing anything - Can be removed
class HelloIter(object):
    def __init__(self):
         self._iter = None
         self.reset()
    
    def reset(self):
        print("\nResetting the iterator ...")
        self._iter = iter([1, 2, 3])
        print("Done ...\n")
    
    @tf.autograph.experimental.do_not_convert  # Not doing anything - Can be removed
    def __call__(self):
        print("Calling Mom ...")
        return next(self._iter)

# The hello function is created as an objected to give it a `state` that can be resetted.
#  Necessary because `structured_function.StructuredFunctionWrapper` actually calls the function
hello = HelloIter() 

for _ in range(3):
    print(hello())

hello.reset()

for _ in range(3):
    print(hello())

tf.data.experimental.enable_debug_mode()

hello_obj = HelloIter()

wrapped_func = structured_function.StructuredFunctionWrapper(
    hello_obj,
    "reduce()",
    input_structure=(),
    # add_to_graph=False,
    use_legacy_function=False
)

hello_obj.reset()  # Must re-initialize the `iter()` because `structured_function.StructuredFunctionWrapper` actually runs the func once

for _ in range(3):
    print(wrapped_func._function())

@tf.autograph.experimental.do_not_convert
def outer_interleave_fn(_):
  return tf.py_function(interleave_fn, [_], Tout=[tf.data.DatasetSpec(tf.TensorSpec(shape=(), dtype=tf.string))])[0]

import tensorflow as tf

N_DATASETS_TO_INTERLEAVE = 10

def hello(idx):
  for j in range(idx):
    yield f"IDX: {idx}"

def make_dataset(idx):
    return tf.data.Dataset.from_generator(
        lambda: hello(idx), output_types=tf.string
    )

datasets = [make_dataset(i) for i in range(N_DATASETS_TO_INTERLEAVE)]

ds = tf.data.Dataset.from_tensor_slices(datasets)
ds = ds.interleave(lambda x: x, cycle_length=N_DATASETS_TO_INTERLEAVE)

for x in ds:
  print(x)