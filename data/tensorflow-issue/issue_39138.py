from tensorflow import keras
from tensorflow.keras import layers

#!/usr/bin/python3
import tensorflow as tf;

def lstm():
  inputs = tf.keras.Input((25,256));
  results = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(units = 512) for i in range(2)])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  m = lstm();
  m.save('lstm.h5');

#!/usr/bin/python3
import tensorflow as tf;

def lstm():
  inputs = tf.keras.Input((25,256));
  results = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units = 512))(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  m = lstm();
  m.save('lstm.h5');

#!/usr/bin/python3
import tensorflow as tf;

def lstm():
  inputs = tf.keras.Input((25,256));
  results = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(units = 512) for i in range(2)])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  m = lstm();
  m.save('lstm.h5');