from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

def conv_block(inputs, conv_type, filter_count, kernel_size, strides, padding='same', relu=True):

  if(conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(filter_count, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(filter_count, kernel_size, padding=padding, strides = strides)(inputs)

  x = tf.keras.layers.BatchNormalization()(x)

  if (relu):
    x = tf.keras.activations.relu(x)

  return x


def _res_bottleneck(inputs, filters, kernel, t, s, r=False):


    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)

    return x

def pyramid_pooling_block(input_tensor, bin_sizes,input_height, input_width):
    # concat_list = []
    width = input_width//32
    height = input_height//32
    # x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (height, width)))(input_tensor)
    concat_list=[input_tensor]
    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(height // bin_size, width // bin_size),
                                             strides=(height // bin_size, width // bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (height, width)))(x)

        concat_list.append(x)
    
    return tf.keras.layers.concatenate(concat_list)



def buildFastScnn(input_height, input_width, input_channel, n_classes, weights_path=None):

    input_layer = tf.keras.layers.Input(shape=(input_height, input_width, input_channel), name ='input_layer')

    lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))

 
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8],input_height, input_width)


    ff_layer1 = conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)


    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)


    classifier = tf.keras.layers.Conv2D(n_classes, (1, 1), padding='same', strides=(1, 1),name = 'Conv2_classifier')(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    
    classifier = tf.keras.layers.Dropout(0.3)(classifier)
    classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
    classifier = tf.keras.activations.softmax(classifier)
    

    fast_scnn = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
    if weights_path is not None:
        fast_scnn.load_weights(weights_path, by_name=True)
    return fast_scnn
    
    
net=buildFastScnn(800, 1600, 3, 20, weights_path=None)
checkpoint = ModelCheckpoint('output_il/weights.{epoch:03d}-{categorical_accuracy:.3f}.h5',
                             monitor='categorical_accuracy',
                             mode='max',
                             verbose=1,save_weights_only=True)
tensorboard = TensorBoard(batch_size=opt.batch_size)
optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
net.fit_generator(train_generator, steps_per_epoch=None, epochs=opt.n_epochs, callbacks=[checkpoint, tensorboard],
                    validation_data=val_generator, validation_steps=None, workers=12,
                    use_multiprocessing=True, shuffle=True, max_queue_size=12, initial_epoch=opt.epoch)