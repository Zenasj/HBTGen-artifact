import numpy as np

model = Sequential()
model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy', 
              metrics=['acc'])

history = model.fit(x=[MODEL_DATA['DATA_TRAIN_PADDED_NORMALIZED']],
                    y=[MODEL_DATA['DATA_LABELS_OHC']],
                    validation_split=0.2,
                    epochs=500,
                    verbose=1)

history = model.fit(x=np.array([MODEL_DATA['DATA_TRAIN_PADDED_NORMALIZED']]),
                    y=np.array([MODEL_DATA['DATA_LABELS_OHC']]),
                    validation_split=0.2,
                    epochs=500,
                    verbose=1)