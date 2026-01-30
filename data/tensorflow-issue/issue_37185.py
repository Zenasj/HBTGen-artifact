model = models.Sequential()
model.add(Conv2D(4, (9, 9), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(8, (9, 9), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (9, 9), activation='relu'))
model.add(MaxPooling2D((2, 2)))


model.add(Flatten())
model.add(Dense(5, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit(ds_train,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=ds_val,
                    validation_steps=validation_steps,
                    epochs=10)