import tensorflow as tf #tf2.0
from collections import OrderedDict

class MyNamedTuple(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(MyNamedTuple, self).__init__(*args, **kwargs)
        self.__dict__ = self
    def __iter__(self):  # iterate over values (not keys)
        for v in self.values():
            yield v
          
class MyModule(tf.Module):
    def __call__(self):
        self.myargs = MyNamedTuple({'arg': 'hello'}) # cause KeyError
        self.v = tf.Variable([1, 2, 3]) # some variables
        
module = MyModule('my_module')
module()
my_vars = module.variables #KeyError