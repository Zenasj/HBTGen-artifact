import random
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow import keras

x = np.random.randn(60,30).astype(np.float32)
y = np.random.randint(low=0, high=10, size = 60).astype(np.int32)

x_tr = x[0:20]
y_tr = y[0:20]
x_val = x[20:40]
y_val = y[20:40]
x_tst = x[40:60]
y_tst = y[40:60]

print(x_tr.shape, y_tr.shape)
print(x_val.shape, y_val.shape)
print(x_tst.shape, y_tst.shape)


tr_dataset = tf.data.Dataset.from_tensor_slices((x_tr,y_tr))
tr_dataset = tr_dataset.batch(batch_size=4).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataset = tr_dataset.batch(batch_size=4).repeat()
tst_dataset = tf.data.Dataset.from_tensor_slices((x_tst,y_tst))
tst_dataset = tst_dataset.batch(batch_size=4)

print(tr_dataset)
print(val_dataset)
print(tst_dataset)

class Model(keras.Model):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.dense = keras.layers.Dense(units=10, activation='softmax')
    def call(self, inputs):
        score = self.dense(inputs)
        return score
  
model = Model(10)
model.compile(optimizer=tf.train.GradientDescentOptimizer(.1),
              loss=keras.losses.sparse_categorical_crossentropy)
model.fit(tr_dataset, epochs=5, steps_per_epoch=20//4,
          validation_data=val_dataset, validation_steps=20//4)

# yhat_from_call_method
sess = keras.backend.get_session()
x_tst_tensor = tf.convert_to_tensor(x_tst)
yhat_from_call_method = sess.run(model(x_tst_tensor))
yhat_from_call_method = np.argmax(yhat_from_call_method, axis = -1)
print(yhat_from_call_method)

# yhat_from_predict_method 
yhat_from_predict_method = model.predict(tst_dataset, steps=20//4)
yhat_from_predict_method = np.argmax(yhat_from_predict_method, axis =-1)
print(yhat_from_predict_method)

print(yhat_from_call_method == yhat_from_predict_method)