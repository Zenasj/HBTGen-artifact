import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), data_format="channels_last", activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(Flatten())
model.add(Dense(len(label), activation='softmax'))
model.summary()

initial_learning_rate = args.lr
    
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate,
decay_steps=4000,
decay_rate=0.96,
staircase=True)

optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model.compile(loss=tf.keras.losses.KLDivergence(), optimizer=optimizer, metrics=['accuracy'])

model_path='./'
file_path = os.path.join(model_path, 'saved-model-{epoch:02d}-{val_loss:.2f}.hdf5') # 
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='max') # 
callback_list = [checkpoint]

hist = model.fit(train_datagen.flow(x_train, y_train, batch_size=args.batch_size), epochs=args.epochs, steps_per_epoch=train_steps, validation_data = (valid_datagen.flow(x_valid, y_valid, batch_size=args.batch_size)), callbacks = callback_list, shuffle=True)

tf.keras.optimizers

eval.py

model = keras.models.load_model(model_dir)
model.summary()

y_pred = model.predict(x_test)