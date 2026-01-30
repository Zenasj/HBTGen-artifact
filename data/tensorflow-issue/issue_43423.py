from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

# Create first model
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Dense(1))
model1.compile()
model1.build([None,3])

# Create second model
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(1))
model2.compile()
model2.build([None,3])


# Concatenate
fusion_model = tf.keras.layers.Concatenate()([model1.output, model2.output])
t = tf.keras.layers.Dense(1, activation='tanh')(fusion_model)
model = tf.keras.models.Model(inputs=[model1.input, model2.input], outputs=t)
model.compile()

#Datasets
ds1 = tf.data.Dataset.from_tensors(([1,2,3],1))
ds2 = tf.data.Dataset.from_tensors(([1,2,3], 2))

print(ds1)
print(ds2)
# Fit
model.fit([ds1,ds2])