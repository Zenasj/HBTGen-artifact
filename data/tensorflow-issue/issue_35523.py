model = Sequential()
model.add(Embedding(vocab_size, dimensions,weights=[embedding_matrix], input_length=pad2, trainable=True))
model.add(GRU(128, return_sequences=True))
model.add(MaxPooling1D())
model.add(AveragePooling1D())
model.add(GRU(128))
model.add(Dense(len(set(outputs)), activation='sigmoid'))
checkpoint = ModelCheckpoint('model-%s-%s-%s-%s' %(setting, indicators, case, str(window_size)), verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# The summary is as expected
train, dev = padded_docs[:split], padded_docs[split:]
y_train, y_dev = y[:split], y[split:]
model.fit(train, y_train, epochs=epochs, verbose=1, callbacks=[checkpoint], batch_size=batch_size, validation_data=(dev, y_dev))
# This is where the script fails