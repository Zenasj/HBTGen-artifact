import numpy as np
from tensorflow.keras import models

class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model # Called by the parent model before training, to inform the callback of what model will be calling it
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs) # Model instance that returns the activations of every layer

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
                raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1] # Obtains the first input sample of the validation data
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w') # Saves arrays to disk
        np.savez(f, activations)
        f.close()

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir', # Location of log files
        histogram_freq=1, # Records activation histogram every 1 epoch
        embeddings_freq=1, # Records embedding data every 1 epoch
    )
]

history = model.fit(x_train, y_train, 
                epochs=20, 
                batch_size=128, 
                validation_split=0.2, 
                callbacks=callbacks)

# Browse to http://localhost:6006 and look at your model training
...

# A list of 2 or more callbacks that can be passed into `model.fit`
callbacks_list = [
        keras.callbacks.EarlyStopping(
                monitor='acc',
                patience=1,
        ),
        
        keras.callbacks.ModelCheckpoint( # Saves the current weights after every epoch
                filepath='my_model.h5',
                monitor='val_loss',
                save_best_only=True, # These two arguments mean you won’t overwrite the model file unless val_loss has improved
        )
]

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])

model.fit(x, y,
                epochs=10,
                batch_size=32,
                callbacks=callbacks_list,
                validation_data=(x_val, y_val)
                )