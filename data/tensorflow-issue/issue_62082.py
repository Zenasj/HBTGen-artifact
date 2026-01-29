# The code snippet in the issue uses custom ExtensionTypes with a bool mask and a tf.Tensor value.
# Inputs are simple scalar tensors as values. Since no explicit batch or spatial shape is given,
# we assume the input is a scalar tensor (shape=()).

import tensorflow as tf
import dataclasses

# Define the ExtensionTypes as provided in the issue but integrated as part of the model submodules
@dataclasses.dataclass
class MaskedTensor(tf.experimental.ExtensionType):
    mask: bool
    value: tf.Tensor

    def __tf_flatten__(self):
        metadata = (self.mask,)  # static config.
        components = (self.value,)  # dynamic values.
        return metadata, components

    @classmethod
    def __tf_unflatten__(cls, metadata, components):
        return cls(*metadata, *components)

@dataclasses.dataclass
class MaskedTensorComp(tf.experimental.ExtensionType):
    mask: bool
    value: tf.Tensor
    mt: MaskedTensor

    def __tf_flatten__(self):
        metadata = (self.mask,)  # static config.
        components = (self.value, self.mt)  # dynamic values.
        return metadata, components

    @classmethod
    def __tf_unflatten__(cls, metadata, components):
        # In actual use, printing inside unflatten can cause TF graph trace issues,
        # so we leave it out or comment.
        # print('Unflattening MaskedTensorComposite', components, metadata)
        return cls(*metadata, *components)

# Because the original issue context is about flatten/unflatten failure,
# we create a Keras Model that accepts a MaskedTensorComp and outputs a simple transformation.
# For demonstration, the model extracts the tensors and outputs the sum of the masked values as a scalar.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights here, just demonstrates processing ExtensionType inputs.

    @tf.function(jit_compile=True)
    def call(self, mt_comp: MaskedTensorComp):
        # Access the components inside MaskedTensorComp
        # mt_comp.mask is bool, mt_comp.value is tensor, mt_comp.mt is MaskedTensor instance
        # mt_comp.mt.value is tensor inside MaskedTensor, mt_comp.mt.mask is bool inside MaskedTensor

        # Convert bool masks to float to perform numeric ops
        mask_val = tf.cast(mt_comp.mask, tf.float32)
        mt_mask_val = tf.cast(mt_comp.mt.mask, tf.float32)

        # Values are tensors (possibly scalar tensors)
        v1 = mt_comp.value
        v2 = mt_comp.mt.value

        # Return a tensor combining the values weighted by masks
        # For example: sum of (mask * value)
        combined = mask_val * v1 + mt_mask_val * v2
        return combined

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a MaskedTensorComp input compatible with MyModel:
    # MaskedTensorComp(mask: bool, value: tensor, mt: MaskedTensor)
    # MaskedTensor(mask: bool, value: tensor)

    # We use scalar tensors as in the original repro code.
    mt0 = MaskedTensor(True, tf.constant(3.0))
    mtc = MaskedTensorComp(False, tf.constant(99.0), mt=mt0)
    return mtc

