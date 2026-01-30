import random
from tensorflow import keras
from tensorflow.keras import layers

emb.shape: TensorShape([2, 3, 180, 320, 32])
cor_l.shape: TensorShape([None, 2, 180, 320])
cor_prob1.shape: TensorShape([2, None, 180, 320])
cor_prob2.shape: TensorShape([2, None, 180, 320, 1])
cor_prob3.shape: TensorShape([2, None, 180, 320, 32])
cor_prob4.shape: TensorShape([2, 180, 320, None, 32])
cor_prob5.shape: TensorShape([2, 180, 320, None])
aligned_fea.shape: TensorShape([2, 180, 320, 3, 32])
aligned_fea.shape: TensorShape([2, 180, 320, 96])

emb.shape: TensorShape([2, 3, 180, 320, 32])
cor_l.shape: TensorShape([3, 2, 180, 320])
cor_prob1.shape: TensorShape([2, 3, 180, 320])
cor_prob2.shape: TensorShape([2, 3, 180, 320, 1])
cor_prob3.shape: TensorShape([2, 3, 180, 320, 32])
cor_prob4.shape: TensorShape([2, 180, 320, 3, 32])
cor_prob5.shape: TensorShape([2, 180, 320, 96])
aligned_fea.shape: TensorShape([2, 180, 320, 3, 32])
aligned_fea.shape: TensorShape([2, 180, 320, 96])

python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import tensorflow as tf

class TestModel(tf.keras.Model):

    def __init__(self, N):
        super(TestModel, self).__init__()
        self.tAtt_1 = tf.keras.layers.Conv2D(4, (3, 3), (1, 1), "same")
        self.tAtt_2 = tf.keras.layers.Conv2D(4, (3, 3), (1, 1), "same")
        self.nframes = N
        self.center = self.nframes // 2

    def __call__(self, aligned_fea):

        aligned_fea_shape = tf.shape(aligned_fea) # B, N, H, W, C
        B = aligned_fea_shape[0]
        N = aligned_fea_shape[1]
        H = aligned_fea_shape[2]
        W = aligned_fea_shape[3]
        C = aligned_fea_shape[4]

        # for i in range(self.nframe):
        #     aligned_fea = aligned_fea.write(i, x)
        #     tf.print("alienged_fea:", aligned_fea.size())

        emb_ref = self.tAtt_1(aligned_fea[:, self.center, :, :, :])  # B, H, W, C
        emb = tf.reshape(aligned_fea, [-1, H, W, C])  # BN, H, W, C
        emb = self.tAtt_2(emb)
        emb = tf.reshape(emb, [B, N, H, W, -1])

        cor_l = tf.TensorArray(dtype=tf.float32, size=N) # TENSORFLOW BUG HERE. REPLACE N with self.nframes, everything will be OK

        def cond(i, N, input, arr):
            return tf.less(i, N)

        def body(i, N, input, arr):
            emb_nbr = input[:, i, :, :, :]
            cor_tmp = tf.reduce_sum(emb_nbr * emb_ref, axis=3)  # B, H, W
            arr = arr.write(i, cor_tmp)
            i = tf.add(i, 1)
            return i, N, input, arr

        _, _, _, cor_l = tf.while_loop(cond, body, [0, N, emb, cor_l])  # N * (B, H, W)

        tf.print("aliged_fea shape:", cor_l.size())

        t = cor_l.stack()
        tf.print("Stack tensor shape:", t.shape)

        return t


gpu_list = [f"/gpu:{i}" for i in range(2)]
mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpu_list, cross_device_ops=tf.distribute.NcclAllReduce())
nframes = 5

def train_step(model, x):
    output = model(x)

@tf.function
def multigpu_train_step(model, x):
    mirrored_strategy.run(train_step, args=(model, x))

values = tf.random.normal([16,5,32,32,3])
sample_dataset = tf.data.Dataset.from_tensor_slices(values)
sample_dataset = sample_dataset.batch(4)
with mirrored_strategy.scope():
    model = TestModel(nframes)
    sample_dataset = mirrored_strategy.experimental_distribute_dataset(sample_dataset)
    for x in sample_dataset:
        multigpu_train_step(model, x)

N = aligned_fea.shape[1]
if N is None:
  # Dynamc shape, revert to tf.shape; N will be a Tensor
  N = tf.shape(aligned_fea)[1]
# Static shape, N is a Python int

cor_l = []
for i in range(self.nframes):
    emb_nbr = emb[:, i, :, :, :]
    cor_tmp = tf.reduce_sum(emb_nbr * emb_ref, axis=3)  # B, H, W
    cor_l.append(cor_tmp)
cor_l = tf.stack(cor_l)  # N, B, H, W