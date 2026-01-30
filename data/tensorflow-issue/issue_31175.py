py
import tensorflow as tf

@tf.function
def run_epoch(dataset, step):
    print('retrace')
    for X in dataset:
        step(X)

class Model:
    def step(self, X):
        return X * 2

dataset = tf.data.Dataset.from_tensor_slices(list(range(128)))
model = Model()

for i in range(20):
    # leads to retrace of run_epoch due to non-identity model.step
    # (i.e. `model.step is not model.step`)
    run_epoch(dataset, model.step)

for i in range(20):
    # passing the entire `model` avoids the retrace.
    run_epoch(dataset, model)

py
import timeit, types, tensorflow as tf

class Model:
  def step(self):
    pass

m = Model()
timeit.timeit('m.step is m.step', number=1_000_000, globals={'m': m})
timeit.timeit('m.step == m.step', number=1_000_000, globals={'m': m})
# still fairly fast
timeit.timeit('var == var if type(var) == types.MethodType else var is var', number=1_000_000, globals={'var': m.step, 'types': types})

@tf.function
def graph(model):
  pass

timeit.timeit('graph(m)', number=1_000, globals={'graph': graph, 'm': m})