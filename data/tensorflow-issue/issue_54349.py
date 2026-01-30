import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_and_configure_model(optimizer, loss, metrics, path):
  model = ResNet50V2(include_top=True, weights='imagenet')
  transfer_layer = model.get_layer('avg_pool')
  resnet_submodel = Model(inputs=model.input,outputs=transfer_layer.output)
  model_config = resnet_submodel.get_config()
  
  submodel = model_config['layers']
  submodel.remove(submodel[0]) # Remove the previous input layer
  
  input_layer = keras.Input(shape=(224, 224, 3), dtype='float32',name="input") # Create a new input layer
  normalization = tf.keras.layers.Normalization(mean=[118.662, 119.194, 96.877], variance=[2769.232, 2633.742, 2702.492], axis=-1, dtype='float32')(input_layer)
  rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1, dtype='float32')(normalization)  
  
  new_model = Model(inputs=input_layer,outputs=rescaling) # Declare pre-processing model to be merged with the ResNet model.
  new_model_cfg = new_model.get_config() 
      
  new_model_cfg['layers'].extend(submodel) # Merge two models.

  # Replace the previous input layer with the output from the preprocessing model
  # (Connect the preprocessing model to the resnet) 
  output_name = new_model_cfg['layers'][2]['name'] # Get the output layer name (rescaling).
  
  new_model_cfg['layers'][3]['inbound_nodes'] = [[[output_name, 0, 0, {}]]] # Replace last inbound node name with the preprocessing model layer name.  
  
  new_model = new_model.__class__.from_config(new_model_cfg, custom_objects={})  # change custom objects if necessary

  # Set back pre-trained weights on new model
  weights = [layer.get_weights() for layer in resnet_submodel.layers[1:]] # For each layer (after the input_layer) in the original resnet50:
  for layer, weight in zip(new_model.layers[3:], weights): # Set imagenet weights on each new model layer.
      layer.set_weights(weight)

  for layer in new_model.layers[:]:
    layer.trainable = False
  for layer in new_model.layers[:]:      
    trainable = True
    layer.trainable = trainable # Train everything. 

  transfer_layer = new_model.get_layer('avg_pool')
  #dropout = tf.keras.layers.Dropout(rate=0.3)(transfer_layer.output)
  species = Dense(1000, activation='softmax', dtype='float32',name='species')(transfer_layer.output) # Specify dtype for handling mixed precision specifications.

  model = keras.Model(
      inputs=[new_model.inputs],
      outputs=[species],
  )
  if not path == None :
    model.load_weights(path)
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model

normalization = tf.keras.layers.Normalization(mean=[118.662, 119.194, 96.877], variance=[2769.232, 2633.742, 2702.492], axis=-1)(input_layer)

rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)(normalization)
model_2 = Model(inputs=input_layer,outputs=rescaling)

pred = model_2.predict(img)
pred = tf.cast(pred, tf.uint8)
pred = tf.squeeze(pred,axis=0)
pred = tf.io.encode_jpeg(pred)

fname = tf.constant('norm_then_scale.jpg')
fwrite = tf.io.write_file(fname, pred)