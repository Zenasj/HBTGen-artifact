import tensorflow as tf

BATCH_SIZE = 64
IMG_SIZE = 98 
N_CLASSES = 179   
N_EPOCHS = 5

model = Sequential([
        Conv2D(filters=32, kernel_size=3, strides=1, activation=tf.nn.relu, input_shape=(IMG_SIZE, IMG_SIZE, 1), padding="same"),
	AveragePooling2D(pool_size=(2, 2), strides=2, padding="same"),
	Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, padding="same"),
	AveragePooling2D(pool_size=(2, 2), padding="same"),
	Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, padding="same"),
	AveragePooling2D(pool_size=(2, 2), strides=2, padding="same"),
	Flatten(),
	Dense(units=1024, activation=tf.nn.relu),
	Dropout(rate=0.8),
	Dense(units=N_CLASSES, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_data = val_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_data = test_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

train_label = to_categorical(train_label, N_CLASSES)
val_label = to_categorical(val_label, N_CLASSES)
test_label = to_categorical(test_label, N_CLASSES)
model.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(val_data, val_label))