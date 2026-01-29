# tf.random.uniform((B=1, H=2, W=2, C=1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils

# Custom BatchNorm layer adapted from the issue to avoid incompatibility of tf.keras.layers.BatchNormalization with TFLite.
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 trainable=True,
                 virtual_batch_size=None,
                 name=None,
                 **kwargs):
        super(BatchNorm, self).__init__(
            name=name, **kwargs)
        if isinstance(axis, list):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('axis must be int or list, type given: %s' % type(axis))
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.virtual_batch_size = virtual_batch_size

        self.fused = True
        self._bessels_correction_test_only = True
        self._trainable_var = None
        self.trainable = trainable

    def build(self, input_shape):
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        if self.virtual_batch_size is not None:
            if self.virtual_batch_size <= 0:
                raise ValueError('virtual_batch_size must be a positive integer that divides the true batch size of the input Tensor')
            # If using virtual batches, the first dimension must be the batch dimension and cannot be the batch norm axis
            if 0 in self.axis:
                raise ValueError('When using virtual_batch_size, the batch dimension must be 0 and thus axis cannot include 0')

        if self.fused:
            if self.axis == [1]:
                self._data_format = 'NCHW'
            elif self.axis == [3]:
                self._data_format = 'NHWC'
            else:
                raise ValueError('Unsupported axis, fused batch norm only supports axis == [1] or axis == [3]')

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input shape: ', input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)]
            if self.virtual_batch_size is not None:
                # When using virtual batches, add an extra dim at index 1
                param_shape.insert(1, 1)
                for idx, x in enumerate(self.axis):
                    self.axis[idx] = x + 1  # Account for added dimension

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            if self.fused:
                self._gamma_const = K.constant(1.0, dtype=self._param_dtype, shape=param_shape)

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.beta = None
            if self.fused:
                self._beta_const = K.constant(0.0, dtype=self._param_dtype, shape=param_shape)

        # Disable variable partitioning when creating the moving mean and variance
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_mean_initializer,
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.MEAN,
            experimental_autocast=False)

        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_variance_initializer,
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.MEAN,
            experimental_autocast=False)
        self.built = True

    def call(self, inputs, training=None):
        training = self._get_training_value(training)
        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training)
            return outputs
        # Fallback (not implemented here) for non-fused path could be added if needed.

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()

        if isinstance(training, int):
            training = bool(training)
        return training

    def _fused_batch_norm(self, inputs, training):
        """Returns the output of fused batch norm."""
        beta = self.beta if self.center else self._beta_const
        gamma = self.gamma if self.scale else self._gamma_const

        def _fused_batch_norm_training():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                epsilon=self.epsilon,
                data_format=self._data_format)

        def _fused_batch_norm_inference():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format)

        if training:
            output, mean, variance = _fused_batch_norm_training()
        else:
            output, mean, variance = _fused_batch_norm_inference()

        training_value = tf_utils.constant_value(training)
        momentum = self.momentum if (training_value is None or training_value) else 1.0

        def mean_update():
            return self._assign_moving_average(self.moving_mean, mean, momentum)

        def variance_update():
            return self._assign_moving_average(self.moving_variance, variance, momentum)

        if training_value or training_value is None:
            self.add_update(mean_update)
            self.add_update(variance_update)

        return output

    def _assign_moving_average(self, variable, value, momentum):
        decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
            decay = tf.cast(decay, variable.dtype.base_dtype)
        update_delta = (variable - tf.cast(value, variable.dtype)) * decay
        return tf.compat.v1.assign_sub(variable, update_delta)

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            return dtypes.float32
        else:
            return self.dtype or dtypes.float32


# Custom Conv2d + BatchNorm block using the above custom BN instead of tf.keras BatchNormalization
class Conv2d_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, is_use_bias=True, name=None):
        super(Conv2d_BN, self).__init__(name=name)
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                             padding=padding, use_bias=is_use_bias,
                                             kernel_initializer=tf.ones)
        self.bn = BatchNorm(axis=3, name=name + "/bn")

    @tf.function
    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.bn(x)
        return x


# The main model uses a list of these Conv2d_BN layers (with conditional filtering on layer name)
class MyModel(tf.keras.Model):
    def __init__(self, layer_name, layer_filters, name="test_model2"):
        super(MyModel, self).__init__(name=name)
        self.convs = []
        for n, f in zip(layer_name, layer_filters):
            # Skip layers with "2" in name as original code does
            if "2" in n:
                continue
            # Create Conv2d_BN layers with kernel size of 1
            self.convs.append(
                Conv2d_BN(filters=f, kernel_size=1, strides=(1, 1), padding="valid", is_use_bias=False,
                          name=self.name + "/" + n + "/conv1"))
        self.empty_layer = None

    @tf.function
    def call(self, inputs):
        output1 = inputs[0]
        for c_layer in self.convs:
            output1 = c_layer(output1)

        output2 = inputs[1]
        for c_layer in self.convs:
            output2 = c_layer(output2)

        if self.empty_layer is None:
            # Note: In original code this print shows 'None' for diagnostic
            tf.print("None")  # Use tf.print for graph compatibility
        return output1, output2


def my_model_function():
    # Provide sensible layer_name and layer_filters as in original example
    layer_name = ["layer1", "layer2", "layer3", "layer4", "layer5"]
    layer_filters = [3, 4, 5, 6, 7]
    model = MyModel(layer_name, layer_filters)
    # Initialize weights by calling model once
    dummy_input = GetInput()
    model(dummy_input)
    return model


def GetInput():
    # Return a list of two Tensors matching required inputs:
    # Original example uses shape (1,2,2,1) tensors for each input
    # It's a tuple/list of two tensors, one with ones, one with zeros, shape=(1,2,2,1)
    batch_size = 1
    height = 2
    width = 2
    channels = 1
    input1 = tf.ones((batch_size, height, width, channels), dtype=tf.float32)
    input2 = tf.zeros((batch_size, height, width, channels), dtype=tf.float32)
    return [input1, input2]

