from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

class CusModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=2, activation='softmax', name='output')

    def call(self, x):
        return self.dense(x)

dummy_data_x = tf.convert_to_tensor([[0, 0],
                [1, 0],
                [0, 1],
                [1, 1]])
dummy_data_y = tf.convert_to_tensor([0, 1, 0, 1])

model = CusModel()
model.compile(optimizer=tf.keras.optimizers.Adam(10.0),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.evaluate(x=dummy_data_x, y=dummy_data_y))
model.fit(x=dummy_data_x, y=dummy_data_y, epochs=10)
print(model.evaluate(x=dummy_data_x, y=dummy_data_y))
model.save_weights('test_model.weights.h5')

model = CusModel()
model.load_weights('test_model.weights.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(10.0),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.evaluate(x=dummy_data_x, y=dummy_data_y))