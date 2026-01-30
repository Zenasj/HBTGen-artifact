import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# nothing special
# please, the code bellow is not important...

inputs = tf.keras.layers.Input(shape = intput_shape)
x = tf.keras.layers.Dense(128)(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.compile(loss=tf.keras.losses.BinaryCrossEntropy(), optimizer='adam',metrics='acc')
tb_callbacks = tf.keras.callbacks.TensorBoard('./logs/')
model.fit(train_data,validation_set=val_data,epochs=100)
# fitting...
# here, after 30 epochs, val_accuracy is around 95%, I interrupt a training process with stop button in the jupyter notebook
# run again fitting
model.fit(train_data,validation_set=val_data,epochs=100)
# 100% val_accuracy after first epoch

inputs = tf.keras.layers.Input(shape = intput_shape)
x = tf.keras.layers.Dense(128)(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.compile(loss=tf.keras.losses.BinaryCrossEntropy(), optimizer='adam',metrics='acc')
tb_callbacks = tf.keras.callbacks.TensorBoard('./logs/')
model.fit(train_data,validation_set=val_data,epochs=100)
# fitting...
# here, after 30 epochs, val_accuracy is around 95%, I interrupt a training process with stop button in the jupyter notebook
# run again fitting
model.fit(train_data,validation_set=val_data, initial_epoch=31,epochs=100)