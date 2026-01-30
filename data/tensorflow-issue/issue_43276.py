import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Directory where the checkpoints will be saved
checkpoint_dir = '/content/drive/My Drive/HashPro/checkpoints/'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix)

EPOCHS = 50

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback], steps_per_epoch=300)  # Comment to evaluate the model

def build_model(vocab_size, embedding_dim, mid_units, batch_size):
  model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
  tf.keras.layers.Dense(2500, activation='relu'),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(3500, activation='relu'),
  tf.keras.layers.Dense(5500, activation='relu'),
  tf.keras.layers.Dropout(0.15),

  tf.keras.layers.Dense(7500, activation='relu'),
  tf.keras.layers.Dense(9500, activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(mid_units, activation='relu'),
  tf.keras.layers.Dense(mid_units, activation='relu'),
  tf.keras.layers.Dropout(0.15),

  tf.keras.layers.Dense(1500, activation='relu'),
  tf.keras.layers.Dense(500, activation='relu'),
  tf.keras.layers.Dropout(0.15),
  
  tf.keras.layers.Activation('softmax'),
  tf.keras.layers.BatchNormalization()
])
  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

with strategy.scope():
  model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    mid_units=7000,
    batch_size=BATCH_SIZE)

  model.compile(optimizer='Adam', loss=loss)

#Let's see the model's organs!
model.summary()