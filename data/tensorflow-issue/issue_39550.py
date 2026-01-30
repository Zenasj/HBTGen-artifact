from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras import backend as K
import os
import PIL
import csv
import shutil
import numpy as np
import sys
from PIL import Image
from tensorflow.keras import backend as K
import gc

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#*********************** tried this first ***************************************
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# # This is the TPU initialization code that has to be at the beginning.
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
#********************************************************************************

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

print("REPLICAS: ", strategy.num_replicas_in_sync)


from PIL import Image
print(Image.__file__)

# import Image
print(Image.__file__)

def make_generator():

    train_datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, rotation_range=90)
    
    train_generator = train_datagen.flow_from_directory(
        '/content/plant-path/tfdir/train/',  # This is the source directory for training images
        target_size=(400, 400),  # All images will be resized to 150x150
        batch_size=1, 
        
        class_mode='sparse')
    

    return train_generator

def create_model():
         pre_trained_model = InceptionV3(input_shape = (400, 400,3), include_top = False, weights = 'imagenet')

# pre_trained_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(1024, 1024,3))

         for layer in pre_trained_model.layers:
                    if layer.name == 'mixed1':
                                    break
# #     print(layer.name)
                    layer.trainable = False

                    last_layer = pre_trained_model.get_layer('mixed7')
                    last_output = last_layer.output

         from tensorflow.keras.optimizers import RMSprop
         from tensorflow.keras import regularizers
 

         x = Flatten()(last_output)
         x = layers.Dense(1024,  activation= 'relu')(x)
         x = layers.Dropout(.2)(x)
         x = layers.Dense(4, activation= 'softmax')(x)
         modelin = Model(pre_trained_model.input, x)
         return modelin



def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_acc', mode='max')
#     reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save] #, reduce_lr_loss]


batch_size = 456
def scale(image, label):
    image = tf.squeeze(image)
    return image, label  

def get_dataset(i, batchsize = batch_size):
  dataset = tf.data.Dataset.from_generator(make_generator, (tf.float32, tf.float32),
     (tf.TensorShape([400, 400, 3]), tf.TensorShape([None])))
  dataset = dataset.shuffle(buffer_size = 2280)  
  dataset = dataset.cache()
  val = dataset.skip(i*456).take(456).batch(batch_size, drop_remainder=True).prefetch(4)
  
  train = dataset.skip(i*456+456).take(1824).concatenate(dataset.take(456*i)).batch(batch_size, drop_remainder=True).prefetch(15)
  return train, val

for i in range(5):
  
  #********* this was created to trouble shoot but did not resolve issue
  # dataset = tf.data.Dataset.from_generator(make_generator, (tf.float32, tf.float32),
  #    (tf.TensorShape([1, 400, 400, 3]), tf.TensorShape([None])))
  # dataset = dataset.shuffle(buffer_size = 2280)
    
    # val = dataset.skip(i*456).take(456).batch(batch_size, drop_remainder=True).prefetch(4)
  
    # train = dataset.skip(i*456+456).take(1824).concatenate(dataset.take(456*i)).batch(batch_size, drop_remainder=True).prefetch(15)  

  #******************************************************  
    
    train, val = get_dataset(i) 
    
    name_weights = "/content/drive/My Drive/Plant/final_model_fold_D512_I400_mix_1_7_" + str(i) + ".{epoch:02d}-{val_acc:.2f}.h5"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

    
    with strategy.scope():
       modelinc = create_model()
       modelinc.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])

    modelinc.fit(
                dataset,
                # batch_size = batch_size,
                # steps_per_epoch=1824/batch_size,
                epochs=25,
                # shuffle=True,
                # verbose=1,
                # validation_data = val
                 ) #,
                #  callbacks = callbacks)
    
    print(modelinc.evaluate(val)) 
    K.clear_session()
    
    del name_weights
    del callbacks

    gc.collect()