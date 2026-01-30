from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def prepare_data(x, y):
    x=x.astype('float32')
    y=y.astype('int32')
    # convert from range int[0,255] to float32[-1,1]
    x/=255
    x = 2*x -1
    x=x.reshape((-1,28,28,1))
    y=tf.keras.utils.to_categorical(y,num_classes=10)
    return x, y

# prepare the data
x_train, y_train = prepare_data(x_train, y_train)
x_test, y_test = prepare_data(x_test, y_test)


epochs = 2 #200
batch_size = 256

##simplest model
K.clear_session()
model = tf.keras.Sequential([
    layers.Flatten(),
    layers.Dense(100),
    layers.ReLU(),
    layers.Dense(120),
    layers.ReLU(),
    layers.Dense(10)
]
)

loss_function = tf.losses.CategoricalCrossentropy(from_logits=True)
optimizer = Adam(lr=0.001)
model.compile(loss=loss_function,optimizer=optimizer, metrics=['acc'])


model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))


print("Run evaluate on test dataset")
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
results = model.evaluate(test_ds)
print(results)

##---------------------------
print("Cpy model ")
model_copy= tf.keras.models.clone_model(model)
model_copy.build((None, 28,28,1)) # replace 10 with number of variables in input layer
model_copy.compile(loss=loss_function,optimizer=optimizer, metrics=["accuracy"])
model_copy.set_weights(model.get_weights())

print("Run evaluate with dataset on copied model")
test_ds2 = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
results = model_copy.evaluate(test_ds2)
print(results)