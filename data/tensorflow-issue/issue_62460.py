from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

tf.keras.Model.fit(x=generator)

SparseCategoricalCrossentropy

sparce_categorical_crossentropy

# --------------------------------------------------------------------------------
# CIFAR 10
# --------------------------------------------------------------------------------
USE_SPARCE_LABEL = True

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# One Hot Encoding the labels when USE_SPARCE_LABEL is False
if not USE_SPARCE_LABEL:
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_validation = keras.utils.to_categorical(y_validation, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


# --------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------
model: Model = Model(
    inputs=inputs, outputs=outputs, name="cifar10"
)

# --------------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------------
if USE_SPARCE_LABEL:
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   # <--- cause incorrect behavior
else:
    loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False)

learning_rate = 1e-3
model.compile(
    optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    loss=loss_fn,     # <---- sparse categorical causes the incorrect behavior
    metrics=["accuracy"]
)

# --------------------------------------------------------------------------------
# Train 
# --------------------------------------------------------------------------------
batch_size = 16
number_of_epochs = 10

def data_label_generator(x, y):
    def _f():
        index = 0
        length = len(x)
        try: 
            while True:                
                yield x[index:index+batch_size], y[index:index+batch_size]
                index = (index + batch_size) % length
        except StopIteration:
            return
        
    return _f

earlystop_callback = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor='val_accuracy'
)

steps_per_epoch = len(y_train) // batch_size
validation_steps = (len(y_validation) // batch_size) - 1  # To avoid run out of data for validation

history = model.fit(
    x=data_label_generator(x_train, y_train)(),  # <--- Generator
    batch_size=batch_size,
    epochs=number_of_epochs,
    verbose=1,
    validation_data=data_label_generator(x_validation, y_validation)(),
    shuffle=True,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_batch_size=batch_size,
    callbacks=[
        earlystop_callback
    ]
)

CategoricalCrossentropy

USE_SPARSE_LABEL=True

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import (
    __version__
)


from keras.layers import (
    Layer,
    Normalization,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Flatten,
    Dropout,
    Reshape,
    Activation,
    ReLU,
    LeakyReLU,
)
from keras.models import (
    Model,
)
from keras.layers import (
    Layer
)
from keras.optimizers import (
    Adam
)
from sklearn.model_selection import train_test_split

print("TensorFlow version: {}".format(tf.__version__))
tf.keras.__version__ = __version__
print("Keras version: {}".format(tf.keras.__version__))

# --------------------------------------------------------------------------------
# CIFAR 10
# --------------------------------------------------------------------------------
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
USE_SPARCE_LABEL = False   # Setting False make it work as expected

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# One Hot Encoding the labels
if not USE_SPARCE_LABEL:
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_validation = keras.utils.to_categorical(y_validation, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# --------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------
inputs = tf.keras.Input(
    name='image',
    shape=INPUT_SHAPE,
    dtype=tf.float32
) 

x = Conv2D(                                           
    filters=32, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="same",
    activation='relu', 
    input_shape=INPUT_SHAPE
)(inputs)
x = BatchNormalization()(x)
x = Conv2D(                                           
    filters=64, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="same",
    activation='relu'
)(x)
x = MaxPooling2D(                                     
    pool_size=(2, 2)
)(x)
x = Dropout(0.20)(x)

x = Conv2D(                                           
    filters=128, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="same",
    activation='relu'
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(                                     
    pool_size=(2, 2)
)(x)
x = Dropout(0.20)(x)

x = Flatten()(x)
x = Dense(300, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.20)(x)
x = Dense(200, activation="relu")(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model: Model = Model(
    inputs=inputs, outputs=outputs, name="cifar10"
)

# --------------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------------
learning_rate = 1e-3

if USE_SPARCE_LABEL:
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
else:
    loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(
    optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    loss=loss_fn,
    metrics=["accuracy"]
)
model.summary()


# --------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------
batch_size = 16
number_of_epochs = 10

def data_label_generator(x, y):
    def _f():
        index = 0
        length = len(x)
        try: 
            while True:                
                yield x[index:index+batch_size], y[index:index+batch_size]
                index = (index + batch_size) % length
        except StopIteration:
            return
        
    return _f

earlystop_callback = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor='val_accuracy'
)

steps_per_epoch = len(y_train) // batch_size
validation_steps = (len(y_validation) // batch_size) - 1  # -1 to avoid run out of data for validation

history = model.fit(
    x=data_label_generator(x_train, y_train)(),
    batch_size=batch_size,
    epochs=number_of_epochs,
    verbose=1,
    validation_data=data_label_generator(x_validation, y_validation)(),
    shuffle=True,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_batch_size=batch_size,
    callbacks=[
        earlystop_callback
    ]
)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import (
    __version__
)


from keras.layers import (
    Layer,
    Normalization,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Flatten,
    Dropout,
    Reshape,
    Activation,
    ReLU,
    LeakyReLU,
)
from keras.models import (
    Model,
)
from keras.layers import (
    Layer
)
from keras.optimizers import (
    Adam
)
from sklearn.model_selection import train_test_split

print("TensorFlow version: {}".format(tf.__version__))
tf.keras.__version__ = __version__
print("Keras version: {}".format(tf.keras.__version__))

# --------------------------------------------------------------------------------
# CIFAR 10
# --------------------------------------------------------------------------------
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
USE_SPARCE_LABEL = False   # Setting False make it work as expected

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# One Hot Encoding the labels
if not USE_SPARCE_LABEL:
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_validation = keras.utils.to_categorical(y_validation, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# --------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------
inputs = tf.keras.Input(
    name='image',
    shape=INPUT_SHAPE,
    dtype=tf.float32
) 

x = Conv2D(                                           
    filters=32, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="same",
    activation='relu', 
    input_shape=INPUT_SHAPE
)(inputs)
x = BatchNormalization()(x)
x = Conv2D(                                           
    filters=64, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="same",
    activation='relu'
)(x)
x = MaxPooling2D(                                     
    pool_size=(2, 2)
)(x)
x = Dropout(0.20)(x)

x = Conv2D(                                           
    filters=128, 
    kernel_size=(3, 3), 
    strides=(1, 1), 
    padding="same",
    activation='relu'
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(                                     
    pool_size=(2, 2)
)(x)
x = Dropout(0.20)(x)

x = Flatten()(x)
x = Dense(300, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.20)(x)
x = Dense(200, activation="relu")(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model: Model = Model(
    inputs=inputs, outputs=outputs, name="cifar10"
)

# --------------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------------
learning_rate = 1e-3

if USE_SPARCE_LABEL:
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
else:
    loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(
    optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    loss=loss_fn,
    metrics=["accuracy"]
)
model.summary()


# --------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------
batch_size = 16
number_of_epochs = 10

def data_label_generator(x, y):
    def _f():
        index = 0
        length = len(x)
        try: 
            while True:                
                yield x[index:index+batch_size], y[index:index+batch_size]
                index = (index + batch_size) % length
        except StopIteration:
            return
        
    return _f

earlystop_callback = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor='val_accuracy'
)

steps_per_epoch = len(y_train) // batch_size
validation_steps = (len(y_validation) // batch_size) - 1  # -1 to avoid run out of data for validation

history = model.fit(
    x=data_label_generator(x_train, y_train)(),
    batch_size=batch_size,
    epochs=number_of_epochs,
    verbose=1,
    validation_data=data_label_generator(x_validation, y_validation)(),
    shuffle=True,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_batch_size=batch_size,
    callbacks=[
        earlystop_callback
    ]
)