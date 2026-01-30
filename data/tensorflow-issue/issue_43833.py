from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
import tensorflow as tf
import cv2
from datetime import datetime

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model

def main(train=True):
    now = datetime.now()
    
    PATH = './my data' # << here is my own data
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)
    
    train_dataset = image_dataset_from_directory(train_dir,
                                                 label_mode='int',
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
        

    validation_dataset = image_dataset_from_directory(validation_dir,
                                                      label_mode='int',
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)


    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE #-1
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    IMG_SHAPE = IMG_SIZE + (3,)

    # base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
    #                                            include_top=False,
    #                                            weights='imagenet')
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                           include_top=False,
                                           weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)

    print ("start!!")
    if train=='true':
        base_model.trainable = True
      
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        
        prediction_layer = tf.keras.layers.Dense(3)
        prediction_batch = prediction_layer(feature_batch_average)

        
        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)     
        # outputs = rescale(x)
        x = base_model(x, training=True)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
        
        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        print (model.summary())    
        print ("len(model.trainable_variables", len(model.trainable_variables))
        
        initial_epochs = 1
        
        loss0, accuracy0 = model.evaluate(validation_dataset)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))
        
        history = model.fit(train_dataset,
                            epochs=initial_epochs,
                            validation_data=validation_dataset)
        
        dt_string = now.strftime("%d%m%Y_%H:%M:%S.hdf5")
        export_path = "./Weight/{}".format(dt_string)
        # model.save(filepath = export_path, save_format='tf')
        # tf.keras.models.save_model(model, export_path, save_format='h5')
        tf.saved_model.save(model, export_path)    
         
    elif train=='lite':
        load_path = './Weight/08102020_09:04:46/'
        converter = tf.lite.TFLiteConverter.from_saved_model(load_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        open("./converted_model.tflite", "wb").write(tflite_model)    
        
        
if __name__ == "__main__":
    main(train='lite')