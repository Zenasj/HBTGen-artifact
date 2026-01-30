import numpy as np
import random

class DataGenerator(Sequence):
    def __init__(self, split):
        self.split = split
        
    def __len__(self):
        return 2

    def __getitem__(self, index):

        print (f'\n split: {self.split} generator, index: {index}', flush=True)
        y = np.random.uniform(low=0, high=1, size=(2_000, 1))
        y = (y > 0.5).astype(np.int32)
        X = np.random.normal(loc=0, scale=1, size=(2_000, 10))
        X = X.astype(np.float32)
        return X, y

    def on_epoch_end(self):
        print (f'on epoch end: {self.split}', flush=True)

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(10,)))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

training_generator = DataGenerator(
    split='training')

validation_generator = DataGenerator(
    split='validation')

model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    use_multiprocessing=False,
    max_queue_size=10,
    epochs=4,
    shuffle=False,
    workers=0,
    verbose=0)