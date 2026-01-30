from tensorflow.keras import optimizers

import tensorflow as tf
from  tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
#-----------------------------------------------------##
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)))

model.add(layers.Conv2D(32, (3, 3), padding='same'))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.UpSampling2D((2, 2), interpolation='nearest'))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model_dir = '/home/nymble'
nymble_model = models.load_model(os.path.join(model_dir,'sample_model.h5'))
print("Model loaded")

converter = tf.lite.TFLiteConverter.from_keras_model(nymble_model)
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflmodel = converter.convert()