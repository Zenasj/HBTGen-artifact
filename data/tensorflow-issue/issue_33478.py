import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.Sequential([
  tf.keras.layers.Dense(50, input_shape=(1,)),
  tf.keras.layers.Dense(1),
])

def get_call_fn(layer: tf.keras.layers.Layer) -> Callable[[tf.Tensor], tf.Tensor]:
  old_call_fn = layer.call
  def call(input: tf.Tensor) -> tf.Tensor:
    output = old_call_fn(input)
    for hook in layer._hooks:
        hook_result = hook(input, output)
        if hook_result is not None:
          output = hook_result
    return output
  return call

for layer in model.layers:
  layer._hooks = []
  layer.call = get_call_fn(layer)
  layer.register_hook = lambda hook: layer._hooks.append(hook)

class InputOutputSaver:
  def __call__(self, input: tf.Tensor, output: tf.Tensor) -> None:
    self.input = input
    self.output = output

savers = {}
for layer in model.layers:
  saver = InputOutputSaver()
  layer.register_hook(saver)
  savers[layer] = saver

inps = tf.convert_to_tensor([[1.], [2.], [3.],])
print(model(inps))
print(savers[model.layers[0]].output)

original_model = tf.keras.Model(inputs, outputs)
activation_outputs = [layer.output for layer in original_model.layers]
activation_model = tf.keras.Model(inputs, activation_outputs)

seq_1 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128),
])
seq_2 = tf.keras.models.Sequential([
  tf.keras.layers.ReLU(),
  tf.keras.layers.Dense(10)
])
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
images, labels = next(iter(train_ds))
with tf.GradientTape() as tape:
  out = seq_1(images)
  predictions = seq_2(out)
  loss = loss_object(labels, predictions)
  
print(tape.gradient(loss, out))

input_1 = tf.keras.Input((5,), name='input_1')
dense = tf.keras.layers.Dense(units=3)
output_1 = dense(input_1) 
print(dense.inbound_nodes[0].input_tensors)  #<tf.Tensor 'input_1:0' shape=(None, 5) dtype=float32>
input_2 = tf.keras.Input((5,), name='input_2')
output_2 = dense(input_2)
print(dense.inbound_nodes[1].input_tensors)  #<tf.Tensor 'input_2:0' shape=(None, 5) dtype=float32>

def proxy_call(input:tf.Tensor, obj:tf.keras.layers.Layer)->tf.Tensor:
    if obj._before_call is not None:
        obj._before_call(obj,input)
    output = obj._old_call(input)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, input, output)
        if hook_result is not None:
            output = hook_result
    return output

def hook_layer_call(layers: List[tf.keras.layers.Layer], 
                    before_call: Callable[[tf.keras.layers.Layer, tf.Tensor],None]=None, 
                    after_call: Callable[[tf.keras.layers.Layer, tf.Tensor, tf.Tensor],Optional[tf.Tensor]]=None):
    for layer in layers:
        layer._before_call = before_call
        layer._after_call = after_call
        layer._old_call = layer.call
        layer.call = functools.partial(proxy_call, obj=layer)

def print_input_output(layer:tf.keras.layers.Layer, input:tf.Tensor, output:tf.Tensor):
    print(input, output)

# suppose you have a model(such as a tf.keras.Sequential instance)
hook_layer_call(model.layers, after_call=print_input_output)