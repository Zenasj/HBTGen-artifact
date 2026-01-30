from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
print('Using Tensorflow version {} (git version {})'.format(tf.version.VERSION, tf.version.GIT_VERSION))
import numpy as np
from tensorflow.data import Dataset
from tensorflow.feature_column import numeric_column
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import DenseFeatures, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

def make_model():
    fc1 = numeric_column('fc_name')
    dict_input = {'fc_name': Input(1)}
    
    out = DenseFeatures(fc1)(dict_input)
    out = Dense(5, name='dense_feature1')(out)
    out = Dense(1, name='dense_ouput')(out)
    
    return Model(inputs=dict_input, outputs=out)

array = np.ones((1000,1), dtype=np.float)
array_target = np.ones((1000,1), dtype=np.float)

batch_size = 4
dict_array = {'teddy_bear': array}
input_dataset = Dataset.from_tensor_slices(dict_array).batch(batch_size)
target_dataset = Dataset.from_tensor_slices(array_target).batch(batch_size)
complete_dataset = Dataset.zip((input_dataset, target_dataset)).shuffle(10000)

model = make_model()
#model.summary()
for x, y in complete_dataset.take(1):
    print(model(x))
    
loss_fn = MeanSquaredError()
optimizer = Adam(learning_rate=1e-3)

@tf.function
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = loss_fn(target, outputs)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

EPOCHS = 5
loss = 0
for epoch in range(EPOCHS):
    for x, y in complete_dataset:
        loss = train_step(x, y)
    print('Epoch n°{:d}, loss = {:5.4f}'.format(epoch + 1, loss))
for x, y in complete_dataset.take(1):
    print(model(x))
    
model = make_model()
model.summary()
model.compile(optimizer, loss_fn)
model.fit(complete_dataset, epochs=EPOCHS)
for x, y in complete_dataset.take(1):
    print(model(x))