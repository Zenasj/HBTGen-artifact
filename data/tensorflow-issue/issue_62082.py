import dataclasses
import tensorflow as tf

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
        metadata = (self.mask)  # static config.
        components = (self.value, self.mt)  # dynamic values.
        return metadata, components

    @classmethod
    def __tf_unflatten__(cls, metadata, components):
        print('Unflattening MaskedTensorComposite', components, metadata)
        return cls(*metadata, *components)

mt0 = MaskedTensor(True, tf.constant(3.))
mt = MaskedTensorComp(False, tf.constant(99.), mt=mt0)

mt_flat = tf.nest.flatten(mt)
mt_recon = tf.nest.pack_sequence_as(mt, mt_flat)

### Relevant log output