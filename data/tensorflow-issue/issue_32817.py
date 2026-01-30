import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

data = tf.random.normal((60000,30,4))
ground_truth = tf.ones((60000,1))
dataset = tf.data.Dataset.from_tensor_slices((data, ground_truth)).batch(64)

#predefined model here: input: [?, 30,4] output: [?,1]
model.fit(dataset, epochs=5)

'''
    938/Unknown - 16s 17ms/step - loss: 0.02172019-10-07 14:49:49.928619: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
         [[{{node IteratorGetNext}}]]
         [[Shape/_2]]
2019-10-07 14:49:49.928619: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
         [[{{node IteratorGetNext}}]]
938/938 [==============================] - 16s 17ms/step - loss: 0.0217
Epoch 2/5
935/938 [============================>.] - ETA: 0s - loss: 2.2229e-062019-10-07 14:49:59.722216: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
         [[{{node IteratorGetNext}}]]
2019-10-07 14:49:59.722218: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
         [[{{node IteratorGetNext}}]]
         [[Shape/_2]]
'''

import tensorflow as tf
data = tf.random.normal((60000,30,4))
ground_truth = tf.ones((60000,1))
dataset = tf.data.Dataset.from_tensor_slices((data, ground_truth)).batch(64, drop_remainder=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#predefined model here: input: [?, 30,4] output: [?,1]
model.fit(dataset, epochs=5)

# loss, acc = net.evaluate(tst_set)  # do not use this when using a Repeating dataset
loss, acc = net.evaluate(tst_set, steps=3)  # e.g., 3