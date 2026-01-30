from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.python.ops.gen_cudnn_rnn_ops import cudnn_rnn
from tensorflow.python.util.dispatch import add_dispatch_support
from tensorflow.python.keras.layers.core import TFOpLambda

# explicitly adding dispatching works!
# the call to cudnn_rnn_dispatch calls a tfoplayer, which is built CORRECTLY with the
# functional API as the check _in_functional_construction_mode returns true this check
# is done in keras.engine.base_layer and it returns true because we have a tensor t such that:
# t is an instance of <class 'keras.engine.keras_tensor.KerasTensor'> 
# is_instance(t, keras_tensor.KerasTensor) -> True
x = layers.Input([1, 4])
cudnn_rnn_dispatch = add_dispatch_support(cudnn_rnn)
out = cudnn_rnn_dispatch(x, x, 0, tf.zeros(128), rnn_mode='gru')
y = out[0]

# turning into a layer does not work!
# it fails because in the base_layer now the _in_functional_construction_mode check
# erroneuously returns False!
# the _in_functional_construction_mode check now takes place in
# tensorflow.python.keras.engine.base_layer
# and we have the confusing:
# t is an instance of <class 'keras.engine.keras_tensor.KerasTensor'> 
# is_instance(t, keras_tensor.KerasTensor) -> False
x = layers.Input([1, 4])
cudnn_layer = TFOpLambda(cudnn_rnn)
try:
    out = cudnn_layer(x, x, 0, tf.zeros(128), rnn_mode='gru')
    y = out[0]
    print(y)
except TypeError:
    print('IM BROOOOOOOOOOOOOOOKEN')