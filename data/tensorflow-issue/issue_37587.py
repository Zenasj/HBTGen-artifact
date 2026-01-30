class MyCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        super(MyCallback, self).__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--------- pre-predict stop_training={self.model.stop_training}\n")
        #The problem is in the prediction: if commented ES works fine
        predictions = self.model.predict(self.test_data.batch(512))
        print(f"\n--------- post-predict stop_training={self.model.stop_training}\n")

es = keras.callbacks.EarlyStopping(patience=2)
myc = MyCallback(test_data)

#This causes EarlyStop not to stop
my_callbacks = [es, myc]
#Either of these works fine
#my_callbacks = [myc, es]
#my_callbacks = [es]
...
model.fit(train_data.batch(512),
          validation_data=validation_data.batch(512),
          epochs=100,
          callbacks=my_callbacks,
          verbose=1)