import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def custom_loss_envelop(model_inputs, model_outputs):
    def custom_loss(y_true,y_pred):
        mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
        print()
        print(model_inputs); print()
        print(model_outputs); print()
        dy_dx = keras.backend.gradients(model_outputs, tf.gather(model_inputs, [0], axis=1))
        print(dy_dx); print()
        d2y_dx2 = keras.backend.gradients(dy_dx, tf.gather(model_inputs, [0], axis=1))
        print(d2y_dx2); print()

        r = tf.multiply(model_outputs, tf.gather(dy_dx, [0], axis=1)) - tf.multiply(tf.gather(model_inputs, [1], axis=1), tf.gather(d2y_dx2, [0], axis=1)) # y*dy_dx[0] - x[1]*d2y_dx[0]2

        r = keras.backend.mean(keras.backend.square(r))
        loss = mse_loss + r
        return loss
    return custom_loss

nx=100;
inputs_train=np.random.uniform(0,1,(nx,2)); outputs_train=np.random.uniform(0,1,(nx,1))
inputs_val=np.random.uniform(0,1,(int(nx/2),2)); outputs_val=np.random.uniform(0,1,(int(nx/2),1))
n_hidden_units=50; l2_reg_lambda=0; learning_rate=0.001; dropout_factor=0.0; epochs=3

model = keras.Sequential();
model.add(keras.layers.Dense(n_hidden_units, activation='relu', input_shape=(inputs_train.shape[1],), kernel_regularizer=keras.regularizers.l2(l2_reg_lambda))); #first hidden layer
model.add(keras.layers.Dropout(dropout_factor)); model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(n_hidden_units, activation='relu', kernel_regularizer = keras.regularizers.l2(l2_reg_lambda)));
model.add(keras.layers.Dropout(dropout_factor)); model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(n_hidden_units, activation='relu', kernel_regularizer = keras.regularizers.l2(l2_reg_lambda)));
model.add(keras.layers.Dropout(dropout_factor)); model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(outputs_train.shape[1], activation='linear'));
optimizer1 = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

model.compile(loss=custom_loss_envelop(model.inputs, model.outputs), optimizer=optimizer1, metrics=['mse'])

model.fit(inputs_train, outputs_train, batch_size=100, epochs=epochs, shuffle=True, validation_data=(inputs_val,outputs_val), verbose=1)