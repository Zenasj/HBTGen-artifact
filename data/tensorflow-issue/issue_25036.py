import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        return self.dense(inputs)

model = MyModel()
model.build(input_shape=(None, 1))
model.summary()

if not self._inbound_nodes:
  raise AttributeError('The layer has never been called '
                       'and thus has no defined input shape.')

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
layers = model.layers

for layer in layers:
    name = layer.name
    input_shape = layer.input_shape
    output_shape = layer.output_shape
    print('%s   input shape: %s, output_shape: %s.\n' % (name, input_shape, output_shape))

import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(500,)))
model.add(tf.keras.layers.Dense(32))

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(self._inputs, self._outputs)
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
    def __new__(cls, input_shape, num_classes, neurons=100, layers=2):
        cls._inputs = Input( input_shape, dtype=tf.float32, name='inputs' )
        x = Flatten()(cls._inputs)
        for k in range(layers):
            x =  Dense(neurons)(x)
        cls._outputs = Dense(num_classes)(x)
        
        return super().__new__(cls)
    
model = MyModel( (28,28), 10 )
model.summary()

for HPC in HPCs:
   model = MyModel(**HPC)  # initialize with specific HyperParameterCombination
   model.fit(train_data)
   score = model.evaluate(test_data)

import tensorflow as tf
from tensorflow.keras.layers import Input

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        return self.dense(inputs)

    def model(self):
        x = Input(shape=(1))
        return Model(inputs=[x], outputs=self.call(x))

MyModel().model().summary()

class ExampleNetwork(Model):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Sequential([
            Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
            Flatten(),
        ])
        self.concat = Concatenate(axis=-1)
        self.dense = Dense(units=100)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        encoded = self.encoder(inputs[0])
        joined = self.concat([encoded] + inputs[1:])
        return self.dense(joined)

model = ExampleNetwork()
first_batch = [tf.zeros((1, 64, 64, 3)), tf.zeros((1, 10))]
model(first_batch)
model.summary()
# Model: "example_network_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# sequential_1 (Sequential)    (None, 30752)             896
# _________________________________________________________________
# concatenate (Concatenate)    (None, 30762)             0
# _________________________________________________________________
# dense (Dense)                (None, 100)               3076300
# =================================================================
# Total params: 3,077,196
# Trainable params: 3,077,196
# Non-trainable params: 0
# _________________________________________________________________
print(model.input_shape)
# ListWrapper([TensorShape([None, 64, 64, 3]), TensorShape([None, 10])])

3
model = ClassCNN()
model( tf.random.uniform(shape=(128,28,28,1)) )  # required to call .summary() before .fit()
model.summary()

3
class ClassCNN(tf.keras.Model):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ClassCNN, self).__init__()
        self._input_shape  = input_shape   # = (28, 28, 1)
        self._output_shape = output_shape  # = 10

        self.conv1      = Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv2      = Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu)
        self.maxpool    = MaxPooling2D(pool_size=(2, 2))
        self.dropout1   = Dropout(0.25, name='dropout1')
        self.flatten    = Flatten()
        self.dense1     = Dense(128, activation=tf.nn.relu)
        self.dropout2   = Dropout(0.5, name='dropout2')
        self.activation = Dense(self._output_shape, activation=tf.nn.softmax)

        self.conv1.build(     (None,) + input_shape )
        self.conv2.build(     (None,) + tuple(np.subtract(input_shape[:-1],2)) + (32,) )
        self.maxpool.build(   (None,) + tuple(np.subtract(input_shape[:-1],4)) + (64,) )
        self.dropout1.build( tuple(np.floor_divide(np.subtract(input_shape[:-1],4),2)) + (64,) )
        self.dropout2.build( 128 )
        self.build(           (None,) + input_shape)


    def call(self, x, training=False, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        if training:  x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:  x = self.dropout2(x)
        x = self.activation(x)
        return x

3
class ClassNN(tf.keras.Model):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ClassNN, self).__init__()
        self._input_shape  = np.prod(input_shape)  # = (28, 28, 1) = 784
        self._output_shape = output_shape          # = 10

        self.flatten    = tf.keras.layers.Flatten()
        self.dense1     = tf.keras.layers.Dense(128, activation=tf.nn.relu, )
        self.dropout    = tf.keras.layers.Dropout(0.2)
        self.dense2     = tf.keras.layers.Dense(128, activation=tf.nn.softmax)
        self.activation = tf.keras.layers.Dense(self._output_shape, activation=tf.nn.softmax)

        self.dense1.build( self._input_shape)
        self.dropout.build(128)
        self.build( (None, self._input_shape) )


    def call(self, inputs, training=False, **kwargs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        if training: x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation(x)
        return x

3
def FunctionalCNN(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs, x, name="FunctionalCNN")
    plot_model(model, to_file=os.path.join(os.path.dirname(__file__), "FunctionalCNN.png"))
    return model

3
def SequentialCNN(input_shape, output_shape):
    model = Sequential()
    model.add( Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape) )
    model.add( Conv2D(64, (3, 3), activation='relu') )
    model.add( MaxPooling2D(pool_size=(2, 2)) )
    model.add( Dropout(0.25) )
    model.add( Flatten() )
    model.add( Dense(128, activation='relu') )
    model.add( Dropout(0.5) )
    model.add( Dense(output_shape, activation='softmax') )

    plot_model(model, to_file=os.path.join(os.path.dirname(__file__), "SequentialCNN.png"))
    return model

3
#!/usr/bin/env python3
import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0, 1, 2, 3  # Disable Tensortflow Logging
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )

import tensorflow as tf
import tensorflow.keras as keras
import time

from src.dataset import DataSet
from src.keras.examples.ClassCNN import ClassCNN
from src.keras.examples.ClassNN import ClassNN
from src.keras.examples.FunctionalCNN import FunctionalCNN
from src.keras.examples.SequentialCNN import SequentialCNN
from src.utils.csv import predict_to_csv

tf.random.set_seed(42)

timer_start = time.time()

dataset = DataSet()
config = {
    "verbose":      False,
    "epochs":       12,
    "batch_size":   128,
    "input_shape":  dataset.input_shape(),
    "output_shape": dataset.output_shape(),
}
print("config", config)

# BUG: ClassCNN accuracy is only 36% compared to 75% for SequentialCNN / FunctionalCNN
# SequentialCNN   validation: | loss: 1.3756675141198293 | accuracy: 0.7430952
# FunctionalCNN   validation: | loss: 1.4285654685610816 | accuracy: 0.7835714
# ClassCNN        validation: | loss: 1.9851970995040167 | accuracy: 0.36214286
# ClassNN         validation: | loss: 2.302224604288737  | accuracy: 0.09059524
models = {
    "SequentialCNN": SequentialCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    ),
    "FunctionalCNN": FunctionalCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    ),
    "ClassCNN": ClassCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    ),
    "ClassNN":  ClassNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    )
}


for model_name, model in models.items():
    print(model_name)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    model.fit(
        dataset.data['train_X'], dataset.data['train_Y'],
        batch_size = config["batch_size"],
        epochs     = config["epochs"],
        verbose    = config["verbose"],
        validation_data = (dataset.data["valid_X"], dataset.data["valid_Y"]),
        use_multiprocessing = True, workers = multiprocessing.cpu_count()
    )

for model_name, model in models.items():
    score = model.evaluate(dataset.data['valid_X'], dataset.data['valid_Y'], verbose=config["verbose"])
    print(model_name.ljust(15), "validation:", '| loss:', score[0], '| accuracy:', score[1])

print("time:", int(time.time() - timer_start), "s")