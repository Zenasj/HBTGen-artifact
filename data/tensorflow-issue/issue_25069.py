import tensorflow as tf

import keras.backend.tensorflow_backend as bck
config = bck.tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = bck.tf.OptimizerOptions.ON_1
bck.set_session(bck.tf.Session(config=config))


model = Sequential()

model.add(Conv2D(32, (3,3),padding="same",input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))#64
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(34))#num_classes
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=["accuracy"])


model.summary()
model.get_config()


from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
filepath="save_xla/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"


checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, mode='auto',period=1)

callbacks_list = [csv_log,checkpoint]



hist = model.fit(x, y, batch_size=32,epochs=num_epoch, verbose=1,callbacks=callbacks_list)