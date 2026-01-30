from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

X = tf.constant([[1],
                 [2],
                 [3]], dtype=tf.float32)
y = tf.constant([[5],
                 [4],
                 [6]], dtype=tf.float32)

# y_pred = a * x + b, where weights are intialized as a = 1, b = 0
# thus, MSE = (x - y)**2 / len(x)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=1, kernel_initializer='ones', bias_initializer='zeros')])
model.compile(optimizer='sgd', loss='mean_squared_error')

def mse(y, y_pred):
    assert len(y) == len(y_pred)
    return sum((y - y_pred)**2)/len(y)

print('model.evaluate():')
print('- batch_size=1:', model.evaluate(X, y, batch_size=1, verbose=0))
print('- batch_size=2:', model.evaluate(X, y, batch_size=2, verbose=0))
print('- batch_size=3:', model.evaluate(X, y, batch_size=3, verbose=0))
print()

# incorrect mean of two different-sized batches
# Batch 1 is size 2, but Batch 2 is size 1
# So we should compute a weighted mean, but Tensorflow-Keras fails to do so
print((mse(X[:-1], y[:-1]) + mse(X[-1], y[-1]))/2)