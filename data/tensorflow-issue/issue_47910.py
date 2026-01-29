# tf.random.uniform((B, 1), dtype=tf.float32) ← Assuming input shape is (batch_size, 1) as in example

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.train.experimental import PythonState

class TrackableList(PythonState):
    """
    A trackable Python list that integrates with TensorFlow's tracking
    system to survive serialization/deserialization in SavedModel.
    """
    def __init__(self, initial_list=None):
        # Initialize with a python list or empty list if None
        self._list = initial_list if initial_list is not None else []
        super().__init__()

    def serialize(self):
        # Serialize the list as a Tensor with dtype float64 (to preserve any quantile boundaries)
        # Convert to numpy array of float64, then to tf.Tensor
        return tf.convert_to_tensor(np.array(self._list, dtype=np.float64))

    def deserialize(self, serialized):
        # Deserialize from tf.Tensor to python list
        numpy_arr = serialized.numpy()
        self._list = numpy_arr.tolist()

    def __getitem__(self, idx):
        return self._list[idx]

    def __len__(self):
        return len(self._list)

    def to_list(self):
        return self._list

@tf.keras.utils.register_keras_serializable()
class BucketizeLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    """
    BucketizeLayer with trackable list _boundaries attribute.
    This works properly for saving/loading with SavedModel because
    trackable list is used instead of a plain python list.
    """

    def __init__(self, quantiles, **kwargs):
        super(BucketizeLayer, self).__init__(**kwargs)
        self.quantiles = quantiles
        # _boundaries is a TrackableList that survives serialization
        self._boundaries = TrackableList()

    def adapt(self, data):
        # Adapt by computing quantile boundaries and saving them in the TrackableList
        if isinstance(data, tf.Tensor):
            data = data.numpy()
        boundaries = np.nanquantile(data, self.quantiles).tolist()
        # Update the trackable list's backing python list
        self._boundaries._list = boundaries

    def call(self, data):
        # Convert boundaries to a tf.Tensor to use in math_ops.bucketize
        boundaries_tensor = tf.convert_to_tensor(self._boundaries.to_list(), dtype=data.dtype)
        return math_ops.bucketize(data, boundaries_tensor)

    def get_config(self):
        config = {'quantiles': self.quantiles}
        base_config = super(BucketizeLayer, self).get_config()
        return dict(**config, **base_config)

class MyModel(tf.keras.Model):
    """
    Model encapsulating the custom BucketizeLayer with trackable list boundaries.
    """
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Initialize BucketizeLayer with example quantiles typical in the issue
        self.bucketize_layer = BucketizeLayer(quantiles=[0.1, 0.5, 0.9])
        # Example: initialize boundaries by adapting to some dummy data here, or expect adapt externally.
        # We leave it unadapted for flexibility.

    def adapt(self, data):
        # Expose adapt method for the bucketize layer if needed
        self.bucketize_layer.adapt(data)

    def call(self, inputs):
        # Forward pass applies bucketize_layer
        return self.bucketize_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate random input tensor with shape (batch_size, 1), dtype float32 to match example inputs
    B = 8  # batch size chosen arbitrarily
    return tf.random.uniform((B, 1), dtype=tf.float32)

