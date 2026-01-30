import numpy as np

def representative_data_gen():
    dataset_list = os.listdir(data_dir)
    num_calibration_images = 100
    norm_factor = 255.0
    for i in range(num_calibration_images):
        image_name = next(iter(dataset_list))
 
        image = cv2.imread(os.path.join(data_dir, image_name), 1)
        image = image.astype(np.float32)
        image = image/norm_factor

        image = tf.expand_dims(image, 0)
        yield [image]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]

# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

converter.allow_custom_ops = True
tflite_model = converter.convert()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv2d_block_3layers(input_tensor, n_filters, kernel_size=3, dropout=0.2, 
                         batchnorm=True, activation=True):
    # first layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
                      padding = 'same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    # second layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
                      padding = 'same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    # third layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
                      padding = 'same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    return x

def UNET_v2(nClasses=25, input_height=288, input_width=224, n_filters=64, dropout=0.2, 
            batchnorm=True, activation=True):
  
    img_input = layers.Input(shape=(input_height, input_width, 3))  

    c1 = conv2d_block_3layers(img_input, n_filters * 1, kernel_size=3, batchnorm = batchnorm, activation=activation)
    p1 = layers.MaxPooling2D((2, 2))(c1) 

    c2 = conv2d_block_3layers(p1, n_filters * 2, kernel_size=3, batchnorm = batchnorm,  activation=activation)
    p2 = layers.MaxPooling2D((2, 2))(c2) 

    c3 = conv2d_block_3layers(p2, n_filters * 4, kernel_size=3, batchnorm = batchnorm,  activation=activation)
    p3 = layers.MaxPooling2D((2, 2))(c3) 

    c4 = conv2d_block_3layers(p3, n_filters * 4, kernel_size=3, batchnorm = batchnorm,  activation=activation)
    p4 = layers.MaxPooling2D((2, 2))(c4) 

    c5 = conv2d_block_3layers(p4, n_filters = n_filters * 8, kernel_size=3, batchnorm = batchnorm, activation=activation)
    p5 = layers.Dropout(dropout)(c5) 

    up6  = layers.Conv2DTranspose(n_filters * 4, kernel_size=(3,3), strides=(2,2), padding="same")(p5)
    # up6  = layers.UpSampling2D()(p5) 
    m6 = layers.Concatenate(axis=3)([up6, c4])
    c6 = conv2d_block_3layers(m6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    up7 = layers.Conv2DTranspose(n_filters * 4, kernel_size=(3,3), strides=(2,2), padding="same")(c6)
    # up7  = layers.UpSampling2D()(c6) 
    m7 = layers.Concatenate(axis=3)([up7, c3])
    c7 = conv2d_block_3layers(m7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    up8 = layers.Conv2DTranspose(n_filters * 2, kernel_size=(3,3), strides=(2,2), padding="same")(c7)
    # up8  = layers.UpSampling2D()(c7) 
    m8  = layers.Concatenate(axis=3)([up8, c2])
    c8 = conv2d_block_3layers(m8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    up9 = layers.Conv2DTranspose(n_filters * 1, kernel_size=(3,3), strides=(2,2), padding="same")(c8)
    # up9  = layers.UpSampling2D()(c8) 
    m9 = layers.Concatenate(axis=3)([up9, c1])
    c9 = conv2d_block_3layers(m9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    outputlayer = tf.keras.layers.Conv2D(filters=nClasses, kernel_size=1, activation="softmax")(c9)
    
    model = tf.keras.Model(inputs=img_input, outputs=outputlayer)
    model.summary(line_length=124)
    return model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv2d_block_3layers(input_tensor, n_filters, kernel_size=3, dropout=0.2, 
                         batchnorm=True, activation=True):
    # first layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
                      padding = 'same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    # second layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
                      padding = 'same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    # third layer
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
                      padding = 'same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    return x

def UNET_v2(nClasses=25, input_height=288, input_width=224, n_filters=64, dropout=0.2, 
            batchnorm=True, activation=True):
  
    img_input = layers.Input(shape=(input_height, input_width, 3))  

    c1 = conv2d_block_3layers(img_input, n_filters * 1, kernel_size=3, batchnorm = batchnorm, activation=activation)
    p1 = layers.MaxPooling2D((2, 2))(c1) 

    c2 = conv2d_block_3layers(p1, n_filters * 2, kernel_size=3, batchnorm = batchnorm,  activation=activation)
    p2 = layers.MaxPooling2D((2, 2))(c2) 

    c3 = conv2d_block_3layers(p2, n_filters * 4, kernel_size=3, batchnorm = batchnorm,  activation=activation)
    p3 = layers.MaxPooling2D((2, 2))(c3) 

    c4 = conv2d_block_3layers(p3, n_filters * 4, kernel_size=3, batchnorm = batchnorm,  activation=activation)
    p4 = layers.MaxPooling2D((2, 2))(c4) 

    c5 = conv2d_block_3layers(p4, n_filters = n_filters * 8, kernel_size=3, batchnorm = batchnorm, activation=activation)
    p5 = layers.Dropout(dropout)(c5) 

    up6  = layers.Conv2DTranspose(n_filters * 4, kernel_size=(3,3), strides=(2,2), padding="same")(p5)
    # up6  = layers.UpSampling2D()(p5) 
    m6 = layers.Concatenate(axis=3)([up6, c4])
    c6 = conv2d_block_3layers(m6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    up7 = layers.Conv2DTranspose(n_filters * 4, kernel_size=(3,3), strides=(2,2), padding="same")(c6)
    # up7  = layers.UpSampling2D()(c6) 
    m7 = layers.Concatenate(axis=3)([up7, c3])
    c7 = conv2d_block_3layers(m7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    up8 = layers.Conv2DTranspose(n_filters * 2, kernel_size=(3,3), strides=(2,2), padding="same")(c7)
    # up8  = layers.UpSampling2D()(c7) 
    m8  = layers.Concatenate(axis=3)([up8, c2])
    c8 = conv2d_block_3layers(m8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    up9 = layers.Conv2DTranspose(n_filters * 1, kernel_size=(3,3), strides=(2,2), padding="same")(c8)
    # up9  = layers.UpSampling2D()(c8) 
    m9 = layers.Concatenate(axis=3)([up9, c1])
    c9 = conv2d_block_3layers(m9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, activation=activation)

    outputlayer = tf.keras.layers.Conv2D(filters=nClasses, kernel_size=1, activation="softmax")(c9)
    
    model = tf.keras.Model(inputs=img_input, outputs=outputlayer)
    model.summary(line_length=124)
    return model