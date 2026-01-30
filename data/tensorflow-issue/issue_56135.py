import tensorflow as tf
from tensorflow import keras

def create_model(input_shape, model_output_len):
  tf.keras.backend.clear_session()
  input_layer = Input(shape=(input_shape,1), name='input_layer')

  c1 = Conv1D(filters=128, kernel_size=8, strides=2, padding='same', activation='elu', name='Conv1D_1', \
              kernel_initializer=initializers.glorot_uniform(seed=33), bias_initializer=initializers.HeUniform(seed=33))(input_layer)
  c2 = Conv1D(filters=64, kernel_size=4, strides=1, padding='same', activation='elu', name='Conv1D_2', \
              kernel_initializer=initializers.glorot_uniform(seed=33), bias_initializer=initializers.HeUniform(seed=33))(c1)
  dense = Flatten()(c2)

  dense = Dense(units=512, \
                kernel_initializer=initializers.glorot_uniform(seed=33), bias_initializer=initializers.HeUniform(seed=33), \
                kernel_regularizer=regularizers.L2(0.01))(dense)
  dense = ELU()(dense)
  dense = Dropout(0.33, seed=33)(dense)
  dense = Dense(units=512, activation='elu', \
                kernel_initializer=initializers.glorot_uniform(seed=33), bias_initializer=initializers.HeUniform(seed=33), \
                kernel_regularizer=regularizers.L2(0.01))(dense)
  out = Dense(units=model_output_len, activation='sigmoid', name='output')(dense)

  model = Model(inputs=input_layer, outputs=out)
  return model

model = create_model(X_train_pooled_output.shape[1], y_train.shape[1])

EPOCHS = 1000
BATCH_SIZE = 1024
adam = Adam(learning_rate=0.0003)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True, num_thresholds=5000)])
CALLBACKS_LIST = [earlystop, lrschedule, cb_metrics]
model_hist = model.fit(x=X_train_pooled_output, y=y_train, validation_data=(X_test_pooled_output, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True, callbacks=CALLBACKS_LIST)