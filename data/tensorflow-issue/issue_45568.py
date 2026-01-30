from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.datasets import mnist

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    print("\n\n")
  except RuntimeError as e:
    print(e) 
    
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

(trainX, trainy), (testX, testy) = mnist.load_data()

trainX = trainX.astype('float16') / 255
testX = testX.astype('float16') / 255



inputs = tf.keras.Input(shape=(28, 28, 1), name='digits')
conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name = 'conv_1')
x = conv_1(inputs)
maxPool = tf.keras.layers.MaxPooling2D((2, 2), name = 'MaxPool_1')
x= maxPool(x)
conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name = 'conv_2')
x = conv_2(x)
globMaxPool = tf.keras.layers.GlobalMaxPooling2D()
x = globMaxPool(x)
x = tf.keras.layers.Dense(10)(x)
print('outpout from the Dense layer', x.dtype.name)
outputs = tf.keras.layers.Activation(activation="softmax", dtype = 'float32')(x)
model = tf.keras.Model(inputs, outputs)

print('Outputs dtype: %s' % outputs.dtype.name)
print('Inputs dtype: %s' % inputs.dtype.name)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
initial_weights = model.get_weights()
model.fit(trainX, trainy, epochs = 10)
model.evaluate(testX, testy)