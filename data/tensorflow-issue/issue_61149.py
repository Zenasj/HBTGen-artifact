import tensorflow as tf
from tensorflow import keras

def build_model():
  input = tf.keras.Input([None, None, 3], ragged=True, name="image")
  patches = tf.image.extract_patches(
              images=input,
              sizes=[1, 4, 4, 1],
              strides=[1, 4, 4, 1],
              rates=[1, 1, 1, 1],
              padding="SAME",
          )
  
  return tf.keras.Model(
            inputs=input,
            outputs=patches)
  

model = build_model()