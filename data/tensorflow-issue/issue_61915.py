from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow import keras

if __name__ == '__main__':
  optimizer = keras.optimizers.Adam()
  vh  = keras.Input(shape=(2,3), name = 'vh')
  v1  = keras.layers.Dense(512)(vh)
  
  output  = keras.layers.Dense(1, activation='softmax', name='prediction')(v1)
  
  model = keras.Model(inputs=vh, outputs=[output], name="antibody_model")
  
  model.compile(optimizer=optimizer )
  
  model.save('nn_model.keras')
  test = keras.models.load_model('nn_model.keras')