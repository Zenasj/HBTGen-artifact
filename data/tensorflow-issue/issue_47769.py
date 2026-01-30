import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from os import environ
from tensorflow.config import *
from tensorflow.data import *
from tensorflow.distribute import *
from tensorflow.distribute.cluster_resolver import *
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import Sequence
from tensorflow.tpu.experimental import *
import numpy as np
import tensorflow as tf

def select_strategy():
    gpus = list_physical_devices('GPU')
    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        dev = '/GPU:0' if gpus else '/CPU:0'
        return OneDeviceStrategy(device = dev)
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    tpus = list_logical_devices('TPU')
    return TPUStrategy(resolver)
def make_dataset(seq, sl, bs):
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(sl + 1, drop_remainder = True)
    source = tf.constant(seq, dtype = tf.int32)
    return Dataset.from_tensor_slices(seq) \
                  .window(sl + 1, sl, drop_remainder = True) \
                  .flat_map(flatten_window) \
                  .map(split_input_target) \
                  .batch(bs, drop_remainder = True)
class RandomBatches(Sequence):
    def __init__(self, d, sl, bs):
        self.d = d
        self.sl = sl
        self.bs = bs
        self.batches = self.random_batches()
    def __len__(self):
        return len(self.d) // self.sl // self.bs
    def on_epoch_end(self):
        self.batches = self.random_batches()
    def random_batches(self):
        hi = len(self.d) - self.sl - 1
        return np.random.randint(hi, size = (len(self), self.bs))
    def __getitem__(self, i):
        d, bs, sl = self.d, self.bs, self.sl
        batch = self.batches[i]
        return (np.array([d[s:s + sl] for s in batch]),
                np.array([d[s + 1:s + sl + 1] for s in batch]))

BS, SL = 128, 256
seq = np.random.randint(100, size = 10_000_000)
with select_strategy().scope():
    inp = Input(shape = (None,), batch_size = BS, dtype = tf.int32)
    embedding = Embedding(input_dim = 100, output_dim = 100)
    lstm = LSTM(512, stateful = False, return_sequences = True)
    time_dist = TimeDistributed(Dense(100))
    out = time_dist(lstm(embedding(inp)))
    model = Model(inputs = [inp], outputs = [out])
    loss = SparseCategoricalCrossentropy(from_logits = True)
    opt = RMSprop(learning_rate = 0.004)
    model.compile(optimizer = opt, loss = loss)
    model.summary()
    model.fit(x = RandomBatches(seq, SL, BS), epochs = 100)
    # model.fit(x = make_dataset(seq, SL, BS), epochs = 100)