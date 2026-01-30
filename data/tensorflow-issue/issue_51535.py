import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def create_model():
  max_seq_length = 512
  input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="input_word_ids")
  input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                     name="input_mask")
  input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                      name="input_type_ids")
  
  bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
  pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
  drop = tf.keras.layers.Dropout(0.3)(pooled_output)
  output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(drop)

  model = tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
      },
      outputs= output 
  )

  return model

def train_step(train_batch):
  train_x, train_y = train_batch
  with tf.GradientTape() as tape:
    ypred = model(train_x, training=True)
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(train_y, ypred))
  grads = tape.gradient(loss, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
          loss=tf.keras.losses.BinaryCrossentropy(),
          metrics=[tf.keras.metrics.BinaryAccuracy()])

model.fit(train_data,
          validation_data=valid_data,
          epochs=epochs,
          verbose=1)