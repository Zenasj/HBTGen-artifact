import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

class incremental_learning_withDecreasing_ratio(tf.keras.callbacks.Callback):
    """ Icrementally adjust the Kwta self.ratio attribute every 2 epochs.
         End the learning process when ratio == 0.3  """

    def __init__(self, delta = 0.05):
        super(incremental_learning_withDecreasing_ratio, self).__init__()
        self.delta = delta

    def on_epoch_begin(self, epoch, logs=None):
        # The update occurs at the beginning of every 2 epochs
        if epoch % 2 == 0:  
            for i in range(1, 5): # For each Kwta layer
                name = 'kwta_'+str(i)
                layer = self.model.get_layer(name = name)      
                layer.ratio -= self.delta
            
            print('\n Fine tuning: current ratio {} \n'.format(round(layer.ratio, 2)))
    
    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer('kwta_1')
        if ( round(layer.ratio, 2) == 0.3 ) and epoch % 2 == 1: 
                print('\n Desired Ratio reached, stop training...')
                self.model.stop_training = True

kwta_cnn = tf.keras.models.Sequential([
  layers.Conv2D(32, 3, padding='same', activation=None, input_shape = (32, 32, 3)),
  Kwta(ratio=0.6, conv=True, name='kwta_1'),
  layers.Conv2D(32, 3, padding='same', activation=None),
  Kwta(ratio=0.6, conv=True, name='kwta_2'),
  layers.MaxPooling2D(pool_size=(2,2)),
  layers.Dropout(0.2, seed=42),

  layers.Conv2D(64, 3, padding='same', activation=None),
  Kwta(ratio=0.6, conv=True, name='kwta_3'),
  layers.Conv2D(64, 3, padding='same', activation=None),
  Kwta(ratio=0.6, conv=True, name='kwta_4'),
  layers.MaxPooling2D(pool_size=(2,2)),
  layers.Dropout(0.3, seed=42),

  layers.Flatten(),
  layers.Dense(10, activation='softmax', kernel_regularizer= tf.keras.regularizers.l2(0.0005))
])

history_kwta_ft = kwta_cnn.fit(x=x_train, y=y_train, epochs = 20, callbacks=[incremental_learning_withDecreasing_ratio()])