import math
from tensorflow import keras
from tensorflow.keras import layers

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]  # No of words
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


embed_fr = PositionalEmbedding(vocab_size=fr_vocab_size, d_model=512)
embed_en = PositionalEmbedding(vocab_size=en_vocab_size, d_model=512)
fr = tf.stack(fr_train_sequences[0:2])
en = tf.stack(en_train_sequences[0:2])


print('French input shape:', fr.shape)
fr_emb = embed_fr(fr)
print(fr_emb.shape)

import tensorflow as tf
# Create a tensor
tensor_to_sqrt = tf.constant([4.0, 9.0, 16.0], dtype=tf.float32)

# Take the square root
tensor_sqrt = tf.sqrt(tensor_to_sqrt)

# Print the original tensor and its square root
print("Original Tensor:")
print(tensor_to_sqrt.numpy())

print("\nSquare Root Tensor:")
print(tensor_sqrt.numpy())