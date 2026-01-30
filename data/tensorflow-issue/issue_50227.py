import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = keras.Input(shape=3)
x = SparseConv2D(10)(input)
x = DenseFromSparse()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(sparse_x, y_train, epochs=5)

@tf.function
def dense_from_sparse(sparse_data_batch):
  """Transforms a sparse batch of data into the conventional data batch format
  used in CV applications.
    Args:
      padded_coordinates: A `Tensor` of type `int32`.
        [batch_size, max_num_coords_per_batch, 2], the padded 2D
        coordinates. max_num_coords_per_batch is the max number of coordinates in
        each batch item.
      num_valid_coordinates: A `Tensor` of type `int32`.
        [batch_size], the number of valid coordinates per batch
        item. Only the top num_valid_coordinates[i] entries in coordinates[i],
        padded_features[i] are valid. The rest of the entries
        are paddings.
      padded_features: A `Tensor` of type `float32`.
        [batch_size, max_num_coords_per_batch, in_channels] where
        in_channels is the channel size of the input feature.
    Returns: 
      data_batch: A 'Tensor' of type float32,
      [batch_size, image_width, image_height]"""
  
  indices, num_valid_coordinates, padded_features = sparse_data_batch
  batch_size = padded_features.shape[0]
  sparse_tensors = tf.map_fn(sparse_tensor_fn, tf.range(batch_size))
  dense_batch = tf.map_fn(tf.sparse.to_dense, sparse_tensors)
  return dense_batch


class DenseFromSparse(tf.keras.layers.Layer):
  
  def call(self, inputs):
    return dense_from_sparse(inputs)


@tf.function
def sparse_tensor_fn(i):
  sparse_tensor = tf.sparse.SparseTensor(indices=indices[i][:num_valid_coordinates[i]], 
                    values=tf.squeeze(padded_features[i][:num_valid_coordinates[i]]),dense_shape=dense_shape)
  return sparse_tensor