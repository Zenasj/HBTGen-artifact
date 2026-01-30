BATCH_SIZE = 32
IMG_WIDTH = 320
IMG_HEIGHT = 180

seq = KagglePlanetSequence(CSV_PATH,
                           IMAGES_PATH,
                           im_width=IMG_WIDTH,
                           im_height=IMG_HEIGHT,
                           batch_size=32)

model = models.Sequential()
model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(3, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(seq,
          verbose=1,
          epochs=1,
          workers=4)