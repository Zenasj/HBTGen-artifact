import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

batch_size = 28
epochs = 77
num_classes = 100
import os
save_dir = 'model'
model_name = 'test60_trained_model.h5'
import tensorflow.keras as keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = keras.models.Sequential()
model.add(keras.layers.Conv2DTranspose(filters=11,kernel_size=(19, 19), strides=(19, 19), padding='valid',activation='elu',kernel_initializer='Identity'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_classes))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

batch_size = 28
epochs = 77
num_classes = 100
import os
save_dir = 'model'
model_name = 'test60_trained_model.h5'
import tensorflow.keras as keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
img_rows, img_cols, img_channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = keras.models.Sequential()
model.add(keras.layers.Conv2DTranspose(input_shape = ( img_rows, img_cols, img_channels ), filters = 11 ,kernel_size= 19 , strides= 19 , padding='valid',activation='relu', kernel_initializer = 'Identity'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_classes))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])

class Identity(Initializer):
  """Initializer that generates the identity matrix.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Only usable for generating 2D matrices.

  Examples:

  >>> def make_variable(k, initializer):
  ...   return tf.Variable(initializer(shape=[k, k], dtype=tf.float32))
  >>> make_variable(2, tf.initializers.Identity())
  <tf.Variable ... shape=(2, 2) dtype=float32, numpy=
  array([[1., 0.],
         [0., 1.]], dtype=float32)>
  >>> make_variable(3, tf.initializers.Identity(gain=0.5))
  <tf.Variable ... shape=(3, 3) dtype=float32, numpy=
  array([[0.5, 0. , 0. ],
         [0. , 0.5, 0. ],
         [0. , 0. , 0.5]], dtype=float32)>

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
  """

  def __init__(self, gain=1.0):
    self.gain = gain

  def __call__(self, shape, dtype = dtypes.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
       supported.

    Raises:
      ValueError: If the dtype is not floating point
      ValueError: If the requested shape does not have exactly two axes.
    """
    partition_info = None  # Keeps logic so can be readded later if necessary
    dtype = _assert_float_dtype(dtype)
    full_shape = shape if partition_info is None else partition_info.full_shape

    if len(full_shape) < 2:
      raise ValueError(
          "Identity matrix initializer can only be used for 2D matrices.")
          
    initializer = linalg_ops_impl.eye(full_shape[0],full_shape[1], batch_shape = [full_shape[3],full_shape[2]], dtype = dtypes.float32)
    if partition_info is not None:
      initializer = array_ops.slice(initializer, partition_info.var_offset,
                                    shape)
    return self.gain * initializer

  def get_config(self):
    return {"gain": self.gain}

def eye(num_rows,
        num_columns=None,
        batch_shape=None,
        dtype=dtypes.float32,
        name=None):
  """Construct an identity matrix, or a batch of matrices.

  See `linalg_ops.eye`.
  """
  with ops.name_scope(
      name, default_name='eye', values=[num_rows, num_columns, batch_shape]):
    is_square = num_columns is None
    batch_shape = [] if batch_shape is None else batch_shape
    num_columns = num_rows if num_columns is None else num_columns
     
    # We cannot statically infer what the diagonal size should be:
    if (isinstance(num_rows, ops.Tensor) or
        isinstance(num_columns, ops.Tensor)):
      diag_size = math_ops.minimum(num_rows, num_columns)
    else:
      # We can statically infer the diagonal size, and whether it is square.
      if not isinstance(num_rows, compat.integral_types) or not isinstance(
          num_columns, compat.integral_types):
        raise TypeError(
            'num_rows and num_columns must be positive integer values.')
      is_square = num_rows == num_columns
      diag_size = np.minimum(num_rows, num_columns)

    # We can not statically infer the shape of the tensor.
    if isinstance(batch_shape, ops.Tensor) or isinstance(diag_size, ops.Tensor):
      batch_shape = ops.convert_to_tensor(
          batch_shape, name='shape', dtype=dtypes.int32)
      diag_shape = array_ops.concat((batch_shape, [diag_size]), axis=0)
      
      if not is_square:
        shape = array_ops.concat((batch_shape, [num_rows, num_columns]), axis=0)
    # We can statically infer everything.
    else:
      
      diag_shape = batch_shape + [diag_size]

      if not is_square:
        shape = batch_shape + [num_rows, num_columns]
        
    diag_ones = array_ops.ones(diag_shape, dtype=dtype)
    if is_square:
      return np.array(array_ops.matrix_diag(diag_ones)).reshape(diag_size,diag_size,batch_shape[1],batch_shape[0])
    else:
      zero_matrix = array_ops.zeros(shape, dtype=dtype)
      return np.array(array_ops.matrix_set_diag(zero_matrix, diag_ones)).reshape(num_rows,num_columns,batch_shape[1],batch_shape[0])

# pylint: enable=invalid-name,redefined-builtin