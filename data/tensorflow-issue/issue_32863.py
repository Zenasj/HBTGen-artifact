# tf.random.uniform((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS), dtype=...) 
# Inferred input shape: (batch_size, 128, 128, 1)

import tensorflow as tf

# Constants inferred from code and default FLAGS settings:
IMAGE_HEIGHT = 128
IMAGE_WEITH = 128
NUM_CHANNELS = 1
NUM_LABELS = 4
SEED = 66478
BATCH_SIZE = 32  # Using training batch size from first chunk

def data_type():
    # The original code conditionally used fp16 or fp32.
    # Here, we'll default to float32.
    return tf.float32


def GroupNorm(x, G, eps=1e-05):
    # Group Normalization on NHWC tensor
    # x shape: [N, H, W, C]
    # G: number of groups to split channels into
    N, H, W, C = tf.unstack(tf.shape(x))
    gamma = tf.ones([1, 1, 1, C], dtype=x.dtype)
    beta = tf.zeros([1, 1, 1, C], dtype=x.dtype)
    x = tf.reshape(x, [N, H, W, G, C // G])
    # move channels into last dims for moments calculation with keepdims
    x = tf.transpose(x, [0, 3, 1, 2, 4])  # [N,G,H,W,C//G]
    mean, var = tf.nn.moments(x, axes=[2, 3, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.transpose(x, [0, 2, 3, 1, 4])  # back to [N,H,W,G,C//G]
    x = tf.reshape(x, [N, H, W, C])
    return x * gamma + beta


class ResBlock:
    def __init__(self, stride_num=1, downsample=False):
        # Conv weights shapes:
        # conv1: 1x1 filter, 64 in/out channels
        # conv2: 3x3 filter, 64 in/out channels
        # conv3: 3x3 filter, 64 in/out channels
        dtype = data_type()
        self.conv1_weights = tf.Variable(
            tf.random.truncated_normal([1, 1, 64, 64], stddev=0.1, seed=SEED, dtype=dtype))
        self.conv2_weights = tf.Variable(
            tf.random.truncated_normal([3, 3, 64, 64], stddev=0.1, seed=SEED, dtype=dtype))
        self.conv3_weights = tf.Variable(
            tf.random.truncated_normal([3, 3, 64, 64], stddev=0.1, seed=SEED, dtype=dtype))
        self.stride_num = stride_num
        self.downsample = downsample

    def forward(self, data):
        # Implements:
        # y = x + conv3(...)
        shortcut = data

        # BN + ReLU 1
        out = GroupNorm(data, G=32)
        out = tf.nn.relu(out)

        # Downsample shortcut if needed
        if self.downsample:
            shortcut = tf.nn.conv2d(
                out,
                self.conv1_weights,
                strides=[1, self.stride_num, self.stride_num, 1],
                padding='SAME')

        # conv1 (3x3)
        out = tf.nn.conv2d(
            out,
            self.conv2_weights,
            strides=[1, self.stride_num, self.stride_num, 1],
            padding='SAME')

        # BN2 + ReLU2
        out = GroupNorm(out, G=32)
        out = tf.nn.relu(out)

        # conv2 (3x3)
        out = tf.nn.conv2d(
            out,
            self.conv3_weights,
            strides=[1, 1, 1, 1],
            padding='SAME')

        return shortcut + out


class BRN:
    def __init__(self):
        # Initialize many ResBlocks with specific strides & downsample flags
        # 2 blocks with stride 2 + downsample True for each network branch start,
        # rest stride 1 + no downsample
        dtype = data_type()
        self.ResNet_0_0 = ResBlock(2, True)
        self.ResNet_0_1 = ResBlock(2, True)
        self.ResNet_1_0 = ResBlock(2, True)
        self.ResNet_1_1 = ResBlock(2, True)
        # ResNet_0 to ResNet_21 all stride 1, no downsample
        self.ResNet_0 = ResBlock(1, False)
        self.ResNet_1 = ResBlock(1, False)
        self.ResNet_2 = ResBlock(1, False)
        self.ResNet_3 = ResBlock(1, False)
        self.ResNet_4 = ResBlock(1, False)
        self.ResNet_5 = ResBlock(1, False)
        self.ResNet_6 = ResBlock(1, False)
        self.ResNet_7 = ResBlock(1, False)
        self.ResNet_8 = ResBlock(1, False)
        self.ResNet_9 = ResBlock(1, False)
        self.ResNet_10 = ResBlock(1, False)
        self.ResNet_11 = ResBlock(1, False)
        self.ResNet_12 = ResBlock(1, False)
        self.ResNet_13 = ResBlock(1, False)
        self.ResNet_14 = ResBlock(1, False)
        self.ResNet_15 = ResBlock(1, False)
        self.ResNet_16 = ResBlock(1, False)
        self.ResNet_17 = ResBlock(1, False)
        self.ResNet_18 = ResBlock(1, False)
        self.ResNet_19 = ResBlock(1, False)
        self.ResNet_20 = ResBlock(1, False)
        self.ResNet_21 = ResBlock(1, False)

        self.conv1_weights = tf.Variable(
            tf.random.truncated_normal([3, 3, NUM_CHANNELS, 64], stddev=0.1, seed=SEED, dtype=dtype))

        self.conv2_weights = tf.Variable(
            tf.random.truncated_normal([3, 3, NUM_CHANNELS, 64], stddev=0.1, seed=SEED, dtype=dtype))

        self.fc_weights = tf.Variable(
            tf.random.truncated_normal([64, NUM_LABELS], stddev=0.1, seed=SEED, dtype=dtype))

        self.fc_biases = tf.Variable(
            tf.constant(0.1, shape=[NUM_LABELS], dtype=dtype))

    def forward(self, stft, mfcc):
        # Process stft branch
        out_s = tf.nn.conv2d(stft, self.conv1_weights,
                             strides=[1, 1, 1, 1], padding='VALID')

        out_s = self.ResNet_0_0.forward(out_s)
        out_s = self.ResNet_0_1.forward(out_s)
        out_s = self.ResNet_0.forward(out_s)
        out_s = self.ResNet_2.forward(out_s)
        out_s = self.ResNet_4.forward(out_s)
        out_s = self.ResNet_6.forward(out_s)
        out_s = self.ResNet_8.forward(out_s)
        out_s = self.ResNet_10.forward(out_s)
        out_s = self.ResNet_12.forward(out_s)
        out_s = self.ResNet_14.forward(out_s)
        out_s = self.ResNet_16.forward(out_s)
        out_s = self.ResNet_18.forward(out_s)
        out_s = self.ResNet_20.forward(out_s)

        out_s = GroupNorm(out_s, G=32)
        out_s = tf.nn.relu(out_s)
        out_s = tf.nn.avg_pool(out_s,
                               ksize=[1, tf.shape(out_s)[1], tf.shape(out_s)[2], 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID')

        # Process mfcc branch
        out_m = tf.nn.conv2d(mfcc, self.conv2_weights,
                             strides=[1, 1, 1, 1], padding='VALID')

        out_m = self.ResNet_1_0.forward(out_m)
        out_m = self.ResNet_1_1.forward(out_m)
        out_m = self.ResNet_1.forward(out_m)
        out_m = self.ResNet_3.forward(out_m)
        out_m = self.ResNet_5.forward(out_m)
        out_m = self.ResNet_7.forward(out_m)
        out_m = self.ResNet_9.forward(out_m)
        out_m = self.ResNet_11.forward(out_m)
        out_m = self.ResNet_13.forward(out_m)
        out_m = self.ResNet_15.forward(out_m)
        out_m = self.ResNet_17.forward(out_m)
        out_m = self.ResNet_19.forward(out_m)
        out_m = self.ResNet_21.forward(out_m)

        out_m = GroupNorm(out_m, G=32)
        out_m = tf.nn.relu(out_m)
        out_m = tf.nn.avg_pool(out_m,
                               ksize=[1, tf.shape(out_m)[1], tf.shape(out_m)[2], 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID')

        # Elementwise multiply both branch outputs
        out = tf.multiply(out_s, out_m)

        out_shape = tf.shape(out)
        # Flatten spatial dims and channels for FC layer
        reshape = tf.reshape(out, [out_shape[0], -1])
        # Since fc_weights shape = [64, NUM_LABELS], but flattened input 
        # might not be 64, we infer 64 is the "channel" dimension after pooling
        # But original code uses [64, NUM_LABELS], so we assume flatten output is [batch, 64]
        # To approximate this, add a linear layer to map flatten shape to 64 dims:
        # However, to follow original code exactly, we assume pooled feature map spatial dims reduce to 1x1
        # Thus out shape should be [batch_size, 1, 1, 64] approximately.
        # So flatten reshapes to [batch, 64]. We'll trust pooling size to reduce spatial dims to 1.
        # So just compute matmul as is.
        logits = tf.matmul(reshape, self.fc_weights) + self.fc_biases
        return logits


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate the BRN model (as the core)
        self.brn = BRN()

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs is tuple (stft, mfcc)
        stft, mfcc = inputs
        return self.brn.forward(stft, mfcc)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate random input tensors matching model input expectations:
    # Two inputs: stft and mfcc
    # Both shape: (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS)
    import tensorflow as tf
    stft = tf.random.uniform(
        shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS),
        dtype=data_type(),
        minval=0, maxval=1)
    mfcc = tf.random.uniform(
        shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WEITH, NUM_CHANNELS),
        dtype=data_type(),
        minval=0, maxval=1)
    return (stft, mfcc)

