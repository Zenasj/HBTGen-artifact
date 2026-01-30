class EvalCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for x, y in validation_datasets:
            print(self.model.evaluate(x, y, return_dict=True, verbose=0))

model.fit(..., validation_data=None, callbacks=[EvalCallback()])