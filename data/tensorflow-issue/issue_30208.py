from tensorflow.keras import layers

from tensorflow.python import keras            
model = keras.Sequential()
model.add(keras.layers.Dense(5, input_shape=(4,), activation='sigmoid'))
model.add(keras.layers.Dense(3, input_shape=(5,), use_bias=True))
model.compile('sgd', 'mse')

def extract_inbound_nodes(layer):
     return layer.inbound_nodes if hasattr(layer, 'inbound_nodes') else layer._inbound_nodes

for l_ in model.layers:
   for node_ in extract_inbound_nodes(l_):
       assert isinstance(node_.output_tensors, list)
       assert isinstance(node_.input_tensors, list)
       assert isinstance(node_.output_shapes, list)