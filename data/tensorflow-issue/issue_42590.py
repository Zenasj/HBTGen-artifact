from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

def create_model():
  test_input = tf.keras.Input(shape=(None,), dtype='string', name='test')
  test2_input = tf.keras.Input(shape=(None,), dtype='string', name='test2')
  feature_layer_inputs = {}
  feature_layer_inputs['test'] = test_input
  feature_layer_inputs['test2'] = test2_input

  vocab_list = ['This', 'That', 'Thing']
  feature_col = tf.feature_column.categorical_column_with_vocabulary_list(
      key='test', vocabulary_list=vocab_list,
      num_oov_buckets=0)
  embed_col = tf.feature_column.embedding_column(
      categorical_column=feature_col, dimension=4, combiner='mean')
  first_embed_layer = tf.keras.layers.DenseFeatures(
      feature_columns=[embed_col], name="first_embed_layer")

  second_vocab_list = ['a', 'b', 'c']
  feature_col_two = tf.feature_column.categorical_column_with_vocabulary_list(
      key='test2', vocabulary_list=second_vocab_list,
      num_oov_buckets=0)
  embed_col_two = tf.feature_column.embedding_column(
      categorical_column=feature_col_two, dimension=4, combiner='mean')
  second_embed_layer = tf.keras.layers.DenseFeatures(
      feature_columns=[embed_col_two], name="second_embed_layer")
  
  x = first_embed_layer(feature_layer_inputs)
  y = second_embed_layer(feature_layer_inputs)
  x = tf.keras.layers.concatenate([x, y])
  
  hidden_layer = tf.keras.layers.Dense(units=35, use_bias=False,
      name="user-embeddings-layer")(x)

  model = tf.keras.Model(
    inputs=[v for v in feature_layer_inputs.values()],
    outputs=[hidden_layer]
  )

  model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=.01),
                # loss=loss_func,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  return model

in_tensor = tf.constant(['This', 'That'])
other_tensor = tf.constant(['a', 'b'])

features = {
  'test': in_tensor,
  'test2': other_tensor,
}
y = tf.constant([1, 2])

model = create_model()
history = model.fit(x=features, y=y,
                    epochs=10, shuffle=False, 
                    batch_size=1,
                    verbose=1,
                    callbacks=[])