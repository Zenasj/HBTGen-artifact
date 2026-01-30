import tensorflow as tf

class TypeOverrideSequence(Sequence):

    original_generator = None
    dtype = None

    def __init__(self, generator, dtype= tf.float32 ):
        self.original_generator = generator
        self.dtype = dtype

    def __len__(self):
        return self.original_generator.__len__()

    def on_epoch_end(self):
        self.original_generator.on_epoch_end()

    def __getitem__(self, idx):
        data = self.original_generator.__getitem__(idx)
        return (tf.convert_to_tensor(data[0],dtype=self.dtype), tf.convert_to_tensor(data[1],dtype=self.dtype))