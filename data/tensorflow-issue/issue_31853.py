import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

model = tf.keras.Model(inputs=[input_x], outputs=[logits])
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize; Emphasize not getting false positives with pos_weight
              loss=loss, # tf.nn.weighted_cross_entropy_with_logits(logits,labels,pos_weight=1) # tf.keras.losses.MeanSquaredError()
              # tf.keras.losses.mean_squared_error
              # List of metrics to monitor
              metrics=[tf.keras.losses.MeanSquaredError()])
checkpointer = tf.keras.callbacks.ModelCheckpoint(session_name + '_backup.h5', save_best_only=True, monitor = 'acc', verbose = 0)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3, verbose=1,min_delta=0.005)
history = model.fit(data_train, roi_zoom_train,
                    batch_size=batch_size,
                    epochs = 1,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(data_val, roi_zoom_val),callbacks=[checkpointer,early_stopper]) #
model.save(session_name + '.h5')
model = tf.keras.models.load_model(session_name + '.h5')