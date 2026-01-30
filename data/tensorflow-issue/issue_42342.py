import numpy as np

base_model = ResNet50(weights = None, include_top=False, input_shape=(200, 200, 3))

x = base_model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)

# The model to be trained
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1)]
model.summary()

img_path = 'train/10_right.jpeg'
img = image.load_img(img_path, target_size =(256,256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)