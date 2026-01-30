model = Sequential([
    Conv2D(46, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)),
    Conv2D(46, kernel_size=(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.15),
    
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    Conv2D(256, kernel_size=(3,3), activation='relu'),
    Conv2D(256, kernel_size=(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.5),
    
    Conv2D(32, kernel_size=(3,3), activation='relu'),
    Conv2D(32, kernel_size=(3,3), activation='relu'),
    Conv2D(32, kernel_size=(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.1),
    
    Flatten(),
    Dense(15, activation='softmax', kernel_regularizer=l2(0.16))
])

model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])