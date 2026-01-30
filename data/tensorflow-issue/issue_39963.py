class_weights1 = {0: 1., 1: 50.}

epochs = 1
model = Sequential()
model.add(Conv1D(filters = 32, kernel_size = 2, activation='relu',input_shape=(36,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters = 64, kernel_size = 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.005),loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_Train,Y_Train, epochs = epochs, validation_data =(X_Test,Y_Test), verbose=1,class_weight=class_weights1)