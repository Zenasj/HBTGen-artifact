from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow.keras as keras

print(tf.version.VERSION, tf.executing_eagerly(), keras.layers.BatchNormalization._USE_V2_BEHAVIOR)

# keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False

# tf.compat.v1.disable_eager_execution() # Uncomment this sentence works normally

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

input_shape = (160, 160, 3)
classes = 2000

def preprocess(x, y):
    x = tf.image.resize(x, input_shape[:2])
    if x.shape[2] == 1:
        x = tf.image.grayscale_to_rgb(x)
    x = x / 255

    y = tf.one_hot(0, classes)

    return (x, y), y


base_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=input_shape, classes=classes)
inputs = tf.keras.layers.Input(shape=input_shape)
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(classes)(x)
x = keras.activations.softmax(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x)

model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True),loss="categorical_crossentropy",metrics=['accuracy'])

model.summary()

data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).batch(64).repeat()

model.fit(data_train, steps_per_epoch=10000, epochs=5)