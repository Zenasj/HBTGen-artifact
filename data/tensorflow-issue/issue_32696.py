import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def conv(input, kernel, filt, stride, dilation, pad='same'):
    x = layers.Conv2D(filters=filt, kernel_size=kernel, strides=stride, dilation_rate=dilation, padding=pad, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(input)
    return x

def conv_down(input, filters):
    x = conv(input, 3, filters, 2 ,1)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.LeakyReLU(alpha=0.1)(x) 
    return x

def conv_block(input, filters, stride=1, dilation=2, pad='same', bottleneck=True):
    x = layers.BatchNormalization(axis=-1, fused=True)(input)
    x = layers.LeakyReLU(alpha=0.1)(x) 
    if bottleneck:
        x = conv(x, kernel=1, filt=(filters*4), dilation=dilation, stride=1, pad=pad)
        x = layers.BatchNormalization(axis=-1, fused=True)(x)
        x = layers.LeakyReLU(alpha=0.1)(x) 
    x = conv(x, kernel=3, filt=filters, stride=stride, dilation=dilation, pad=pad)
    return x

def dense_block(x, filters, layers, bottleneck=True):
    x_list = [x]
    for i in range(layers):
        cb = conv_block(x, filters, dilation=2, bottleneck=True)
        x_list.append(cb)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)
        x = attention(x)
    return x

def transition_block(input, filters, att=True):
    x = layers.BatchNormalization(axis=-1, fused=True)(input)
    x = layers.LeakyReLU(alpha=0.1)(x) 
    x = conv(x, kernel=1, filt=filters, stride=1, dilation=2, pad='same')
    x = layers.AveragePooling2D((2,2), strides=(2,2))(x)
    if att:
        x = attention(x)
    return x
    
def attention(input):
    x = channel_att(input)
    x = spatial_att(x)
    return x
    
def channel_att(input, ratio=8):
    channel = input.get_shape()[-1]
    ####
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input)
    avg_pool = tf.keras.layers.Reshape((1,1,channel))(avg_pool)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input)
    max_pool = tf.keras.layers.Reshape((1,1,channel))(max_pool)
    ####
    mlp_0 = layers.Dense(units=channel//ratio, activation=layers.ReLU())
    mlp_1 = layers.Dense(units=channel, activation=layers.ReLU())
    avg_ = mlp_1(mlp_0(avg_pool))
    max_ = mlp_1(mlp_0(max_pool))
    scale = keras.activations.sigmoid(avg_+max_)
    return input*scale

def spatial_att(input, kernel=7):
    avg_pool = tf.math.reduce_mean(input, axis=[3], keepdims=True)
    max_pool = tf.math.reduce_max(input, axis=[3], keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=3)
    concat = layers.Conv2D(filters=1, kernel_size=kernel, padding='same',use_bias=False)(concat)
    concat = keras.activations.sigmoid(concat)
    return input*concat


def create_model():
    Input = layers.Input(shape=(540, 540, 3))
    x = conv(Input, kernel=3, filt=64, stride=1, dilation=2)
    for i in range(8):
        x = dense_block(x, filters=128, layers=3, bottleneck=True)
        x = transition_block(x, filters=128, att=True)
    x = dense_block(x, filters=128, layers=4, bottleneck=True)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = conv(x, kernel=1, filt=45, stride=1, dilation=1)
    model = tf.keras.Model(inputs=Input, outputs=x)
    model.compile(optimizer=keras.optimizers.SGD(), loss=custom_loss)
    return model

with tpu_strategy.scope():
    model=create_model()

model.fit(get_training_dataset(), validation_data=get_validation_dataset(),  initial_epoch=0, steps_per_epoch=steps_per_epoch ,validation_steps=val_steps, epochs=EPOCHS, verbose=1, callbacks=clbk)

tpu=' '
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
tf.config.experimental_connect_to_host(resolver.master())
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)
EPOCHS=250


def conv(input, kernel, filt, stride, dilation, pad='same'):
    x = layers.Conv2D(filters=filt, kernel_size=kernel, strides=stride, dilation_rate=dilation, padding=pad, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(input)
    return x

def conv_down(input, filters):
    x = conv(input, 3, filters, 2 ,1)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.LeakyReLU(alpha=0.1)(x) 
    return x

def conv_block(input, filters, stride=1, dilation=2, pad='same', bottleneck=True):
    x = layers.BatchNormalization(axis=-1, fused=True)(input)
    x = layers.LeakyReLU(alpha=0.1)(x) 
    if bottleneck:
        x = conv(x, kernel=1, filt=(filters*4), dilation=dilation, stride=1, pad=pad)
        x = layers.BatchNormalization(axis=-1, fused=True)(x)
        x = layers.LeakyReLU(alpha=0.1)(x) 
    x = conv(x, kernel=3, filt=filters, stride=stride, dilation=dilation, pad=pad)
    return x

def dense_block(x, filters, layers, bottleneck=True):
    x_list = [x]
    for i in range(layers):
        cb = conv_block(x, filters, dilation=2, bottleneck=True)
        x_list.append(cb)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)
        x = attention(x)
    return x

def transition_block(input, filters, att=True):
    x = layers.BatchNormalization(axis=-1, fused=True)(input)
    x = layers.LeakyReLU(alpha=0.1)(x) 
    x = conv(x, kernel=1, filt=filters, stride=1, dilation=2, pad='same')
    x = layers.AveragePooling2D((2,2), strides=(2,2))(x)
    if att:
        x = attention(x)
    return x
    
def attention(input):
    x = channel_att(input)
    x = spatial_att(x)
    return x
    
def channel_att(input, ratio=8):
    channel = input.get_shape()[-1]
    ####
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input)
    avg_pool = tf.keras.layers.Reshape((-1,1,1,channel))(avg_pool)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input)
    max_pool = tf.keras.layers.Reshape((-1,1,1,channel))(max_pool)
    ####
    mlp_0 = layers.Dense(units=channel//ratio, activation=layers.ReLU())
    mlp_1 = layers.Dense(units=channel, activation=layers.ReLU())
    avg_ = mlp_1(mlp_0(avg_pool))
    max_ = mlp_1(mlp_0(max_pool))
    scale = keras.activations.sigmoid(avg_+max_)
    return input*scale

def spatial_att(input, kernel=7):
    avg_pool = tf.math.reduce_mean(input, axis=[3], keepdims=True)
    max_pool = tf.math.reduce_max(input, axis=[3], keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=3)
    concat = layers.Conv2D(filters=1, kernel_size=kernel, padding='same',use_bias=False)(concat)
    concat = keras.activations.sigmoid(concat)
    return input*concat


def create_model():
    Input = layers.Input(shape=(540, 540, 3))
    x = conv(Input, kernel=3, filt=64, stride=1, dilation=2)
    for i in range(8):
        x = dense_block(x, filters=128, layers=3, bottleneck=True)
        x = transition_block(x, filters=128, att=True)
    x = dense_block(x, filters=128, layers=4, bottleneck=True)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = conv(x, kernel=1, filt=45, stride=1, dilation=1)
    model = tf.keras.Model(inputs=Input, outputs=x)
    model.compile(optimizer=keras.optimizers.SGD(), loss=custom_loss)
    return model


with tpu_strategy.scope():
    model=create_model()


model.fit(get_training_dataset(), validation_data=get_validation_dataset(),  initial_epoch=0, steps_per_epoch=steps_per_epoch ,validation_steps=val_steps, epochs=EPOCHS, verbose=1, callbacks=clbk)

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_host(resolver.master())
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)