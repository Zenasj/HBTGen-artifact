# tf.random.uniform((B,)) where B is batch size (unknown here, single filename tensor scalar of dtype string)
import tensorflow as tf
import scipy.io as sio
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters or layers since the model is just a loader/parser
        # We will encapsulate the logic to load and convert .mat files into tensors

    def call(self, filename):
        # filename is expected to be a scalar tf.string Tensor representing the path to a .mat file

        # Use tf.py_function to wrap the scipy loading operation
        x, label = tf.py_function(self._load_mat_file, [filename], [tf.float32, tf.int8])
        
        # Shapes are unknown but we know from comments:
        # x.shape = (num_timestamp, num_feature)
        # label.shape = (num_timestamp,)
        # Mark shapes as unknown except the rank:
        x.set_shape([None, None])
        label.set_shape([None])

        return x, label

    def _load_mat_file(self, filename_tf):
        # This runs in eager / numpy context.
        filename = filename_tf.numpy().decode('utf-8')  # Convert bytes tensor to string path

        data = sio.loadmat(filename)
        x_np = data['X'].astype(np.float32)  # shape (num_timestamp, num_feature)
        label_np = data['label'].reshape(-1).astype(np.int8)  # shape (num_timestamp,)

        return x_np, label_np

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Per above, model expects a scalar tf.string Tensor representing the filename path
    # For demo, create a dummy string tensor.
    # In practice, datasets are constructed from real .mat file paths (e.g. from tf.data.Dataset.list_files)
    
    # Here just create a dummy filename tensor as placeholder string.
    # Because we cannot create real .mat files here, this is a placeholder:
    return tf.constant("dummy_path_to_file.mat", dtype=tf.string)

