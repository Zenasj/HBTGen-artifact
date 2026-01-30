def regDense(a):
    return layers.Dense(a, activation=LeakyReLU(), kernel_initializer=initializers_v2.HeNormal(),
                       kernel_regularizer=l2(0.001), bias_initializer=initializers_v2.Zeros(),
                       bias_regularizer=l1_l2(0.003, 0.02))


def regLSTM(a):
    return layers.LSTM(a, kernel_regularizer=l1_l2(0.0001, 0.0003),
                kernel_initializer=initializers_v2.GlorotNormal(),
                bias_initializer=initializers_v2.Zeros(),
                return_sequences=True,
                bias_regularizer=l1_l2(0.0002, 0.002))


num_inp = keras.Input(shape=(30, 3, 1), name='nums')
text_inp = keras.Input(shape=(30, 7, 3208), name='text')

embed = layers.Embedding(vocabsize, output_dim=152)(text_inp)
tlstm1 = layers.TimeDistributed(layers.TimeDistributed(regLSTM(256)))(embed)
tdrop1 = layers.Dropout(0.2)(tlstm1)
nlstm1 = layers.TimeDistributed(regLSTM(256))(num_inp)
ndrop1 = layers.Dropout(0.2)(nlstm1)


tlstm2 = layers.TimeDistributed(layers.TimeDistributed(regLSTM(256)))(tdrop1)
tdrop2 = layers.Dropout(0.2)(tlstm2)
nlstm2 = layers.TimeDistributed(regLSTM(256))(ndrop1)
ndrop2 = layers.Dropout(0.2)(nlstm2)


tlstm3 = layers.TimeDistributed(layers.TimeDistributed(regLSTM(256)))(tdrop2)
tdrop3 = layers.Dropout(0.2)(tlstm3)
ndense1 = regDense(213)(ndrop2)
ndrop3 = layers.Dropout(0.5)(ndense1)


tdense1 = regDense(200)(tdrop3)
tdrop4 = layers.Dropout(0.5)(tdense1)
ndense2 = regDense(170)(ndrop3)
ndrop4 = layers.Dropout(0.5)(ndense2)


tdense2 = regDense(144)(tdrop4)
tpool2 = layers.MaxPooling3D((1, 1, 401), padding='same')(tdense2)
trsp1 = layers.Reshape((30, 1152, 7))(tpool2)
tpool3 = layers.MaxPooling2D((1, 3), padding='same')(trsp1)
trsp2 = layers.Reshape((2688, 30))(tpool3)
tpool4 = layers.MaxPooling1D(7, padding='same')(trsp2)
trsp3 = layers.Reshape((30, 384))(tpool4)
tdrop5 = layers.Dropout(0.5)(trsp3)
ndense3 = regDense(128)(ndrop4)
nrsp = layers.Reshape((30, 384))(ndense3)
ndrop5 = layers.Dropout(0.5)(nrsp)


concat = layers.concatenate([tdrop5, ndrop5])
prc1 = regDense(384)(concat)
pdrop1 = layers.Dropout(0.5)(prc1)
prc2 = regDense(192)(pdrop1)
pdrop2 = layers.Dropout(0.5)(prc2)
prc3 = regDense(96)(pdrop2)
pdrop3 = layers.Dropout(0.5)(prc3)
prc4 = regDense(48)(pdrop3)
pdrop4 = layers.Dropout(0.5)(prc4)
prc5 = regDense(24)(pdrop4)
pdrop5 = layers.Dropout(0.3)(prc5)
prc6 = regDense(12)(pdrop5)
lastpool = layers.GlobalAveragePooling1D()(prc6)
last = layers.Dense(1, activation='relu', kernel_regularizer=l2(0.0008),
                    kernel_initializer=initializers_v2.HeNormal(),
                    bias_initializer=initializers_v2.Zeros(),
                    bias_regularizer=l1_l2(0.003, 0.02),
                    name='output')(lastpool)
model = keras.Model(inputs=[num_inp, text_inp], outputs=last)

model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

model.fit(
    {"nums": X1_train, "text": X2_train},
    {"output": y_train}, validation_data=({'nums':X1_test, 'text':X2_test}, y_test),
    epochs=9, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True), callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)],
    batch_size=1)