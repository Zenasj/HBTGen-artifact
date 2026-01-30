from tensorflow import keras
from tensorflow.keras import layers

py
import tensorflow as tf

x = tf.ones([1])
y = tf.ones([1])

@tf.function
def f(x, y):
  return x + y

print(f(x, y).numpy())

# output: array([2.], dtype=float32)

def Encoder():
    inputs = tf.keras.layers.Input(shape=(300,300,3))
    conv1_1 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu")(inputs)
    conv1_2 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu")(conv1_1)

    pool = MaxPool2D(pool_size=(2,2), strides=2)(conv1_2)

    conv2_1 = Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="relu")(pool)
    conv2_2 = Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="relu")(conv2_1)

    pool = MaxPool2D(pool_size=(2,2), strides=2)(conv2_2)

    conv3_1 = Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu")(pool)
    conv3_2 = Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu")(conv3_1)
    conv3_3 = Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu")(conv3_2)

    pool = MaxPool2D(pool_size=(2,2), strides=2)(conv3_3)

    conv4_1 = Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(pool)
    conv4_2 = Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(conv4_1)
    conv4_3 = Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(conv4_2)

    pool = MaxPool2D(pool_size=(2,2), strides=2)(conv4_3)

    conv5_1 = Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(pool)
    conv5_2 = Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(conv5_1)
    conv5_3 = Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(conv5_1)

    # Log Polar Tranform
    lop_polar_feature_maps = Lambda(log_polar_sampling,name="log_polar")(conv5_3)
    

    fc6 = Conv2D(filters=1024,kernel_size=(3,3),padding="same")(lop_polar_feature_maps)
    fc7 = Conv2D(filters=1024,kernel_size=(1,1),padding="same")(fc6)

    # Inverse Log Polar

    conv8_1 = Conv2D(filters=256,kernel_size=(1,1),padding="same")(fc7)
    conv8_2 = Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=2)(conv8_1)
    
    conv9_1 = Conv2D(filters=128,kernel_size=(1,1),padding="same")(conv8_2)
    conv9_2 = Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=2)(conv9_1)
    

    conv10_1 = Conv2D(filters=128,kernel_size=(1,1),padding="same")(conv9_2)
    conv10_2 = Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=1)(conv10_1)
    
    conv11_1 = Conv2D(filters=128,kernel_size=(1,1),padding="same")(conv10_2)
    conv11_2 = Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=1)(conv11_1)



    return tf.keras.Model(inputs=inputs,outputs=conv11_2)