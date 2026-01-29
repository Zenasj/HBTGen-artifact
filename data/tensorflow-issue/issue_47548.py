# tf.constant shape in example is (1, 91) for targ; sample shape (1, 41)
# Inputs to model: shape (batch_size, seq_len=41), dtype int32 → output a,b each (batch_size, 910)
# This adapts code from the issue that involves a model outputting parameters a,b for a Beta-like loss

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.activations as A
import numpy as np

# Custom convolutional residual block for 1D sequences
class ResBlock2(L.Layer):
    def __init__(self, filters, KS=3, conv_type=L.Conv1D, act=A.relu):
        super(ResBlock2, self).__init__()
        self.filters = filters
        self.conv_type = conv_type
        self.norm1 = L.BatchNormalization()
        self.norm2 = L.BatchNormalization()
        self.conv1 = conv_type(filters, KS, 1, padding='SAME', use_bias=False)
        self.conv2 = conv_type(filters, KS, 1, padding='SAME', use_bias=False)
        self.act = act
    
    def build(self, input_shape):
        # Create shortcut conv if filters mismatch, else identity (linear activation)
        if self.filters != input_shape[-1]:
            self.shortcut = self.conv_type(self.filters, 1, 1)
        else:
            self.shortcut = lambda x: x
    
    def call(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(L.Add()([self.shortcut(x), self.norm2(self.conv2(out))]))
        return out

# Custom block that concatenates convolutions with different kernel sizes, applies batchnorm and residual
def CustomBlock(feat_map, rang, feat_out=128, residual=True, pad='SAME', alpha=0.0):
    # rang: tuple like (start, end, step)
    rang = tf.range(*rang)
    out = tf.concat([L.Conv1D(feat_out, int(m), 1, padding=pad)(feat_map) for m in rang], axis=-1)
    out = L.BatchNormalization()(out)
    if residual:
        # shortcut conv to match channels if needed
        if feat_map.shape[-1] != out.shape[-1]:
            shortcut = L.Conv1D(out.shape[-1], 1, 1)
        else:
            shortcut = lambda x: x
        out = L.Add()([shortcut(feat_map), out])
    return A.relu(out, alpha=alpha)

# Upsample by factor 2 along sequence dimension (length)
def upsample(x):
    # x shape: (batch, length, channels)
    # Expand length dimension and tile
    return tf.reshape(tf.tile(tf.expand_dims(x, axis=2), [1,1,2,1]), (-1, 2*x.shape[1], x.shape[-1]))

# Model builder function based on the shared code
def Model_beta(seq_len, AA_types, out_dim, floors=(1,1), 
               floors_eps=1e-7, # Use slightly larger epsilon to avoid graph vs eager issue
               filtfirst=64, rang=(2,10,1), outin=(3,5), filtmid=(150, 200), filtlast=200):
    inp = L.Input((seq_len,), dtype=tf.int32)
    
    # One-hot encode the input sequence
    out = tf.one_hot(inp, AA_types)  # shape (batch, seq_len, AA_types)
    
    # Initial CustomBlock (note: alpha=0.0 means ReLU)
    out = CustomBlock(out, (3,4,1), feat_out=filtfirst, residual=False, alpha=0.0)
    
    outer, inner = outin
    filts = np.linspace(filtmid[0], filtmid[1], outer, dtype='int')
    
    # Stack of upsampling and Residual blocks
    for m in range(outer):
        if m > 0:
            out = upsample(out)
        for n in range(inner):
            filters = filts[m]
            ks = 3
            out = ResBlock2(filters, ks)(out)
    
    # Final CustomBlock without residual
    out = CustomBlock(out, (1,2,1), feat_out=filtlast, residual=False)
    
    # Output layers for parameters a and b of Beta distribution parameters
    a = L.Conv1D(out_dim, 1, 1, activation='relu')(out)
    a = L.GlobalAveragePooling1D()(a)
    a = tf.squeeze(a) + floors[0] + floors_eps  # Ensure output ≥ floors + eps to avoid log(0)
    
    b = L.Conv1D(out_dim, 1, 1, activation='linear')(out)
    b = L.GlobalAveragePooling1D()(b)
    b = tf.squeeze(b) + floors[1] + floors_eps  # Same for b
    
    return tf.keras.Model(inputs=inp, outputs=[a, b])

# Define MyModel class wrapping this model and loss logic
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model parameters: seq_len=41, AA_types=22, output dim=910
        # floors set to 2*(1+eps) as in issue example
        self.inner_model = Model_beta(
            seq_len=41,
            AA_types=22,
            out_dim=910,
            floors=(2,2),
            floors_eps=1e-7
        )
        self.eps = 1e-7  # Epsilon to avoid log(0) in loss
    
    def call(self, inputs):
        # inputs shape: (batch_size, 41), dtype int32
        a, b = self.inner_model(inputs)
        return a, b
    
    @tf.function  # JIT compile compatible
    def neg_log_prob(self, a, b, targ):
        # Custom loss function matching NegLogProb in issue but using eps=1e-7
        # targ shape: (batch_size, 910) with values in [0,1]
        # Compute:
        # -mean(sum( log(a*b) + (a-1)*log(targ+eps) + (b-1)*log(1 - targ^a + eps), axis=-1))
        eps = self.eps
        log_term1 = tf.math.log(a * b)
        log_term2 = (a - 1) * tf.math.log(targ + eps)
        log_term3 = (b - 1) * tf.math.log(1 - tf.math.pow(targ, a) + eps)
        log_sum = log_term1 + log_term2 + log_term3
        sum_log = tf.reduce_sum(log_sum, axis=-1)
        mean_sum_log = tf.reduce_mean(sum_log)
        neg_log_prob = -mean_sum_log
        return neg_log_prob
    
    def compute_loss(self, samples, targ):
        # Run forward and compute loss for training step
        a, b = self(samples)
        loss = self.neg_log_prob(a, b, targ)
        return loss, (a, b)

def my_model_function():
    # Returns a new instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor compatible with MyModel
    # Input shape: (batch_size, 41), dtype int32, values in [0,21] (22 types)
    batch_size = 2
    seq_len = 41
    AA_types = 22
    samples = tf.random.uniform(shape=(batch_size, seq_len), minval=0, maxval=AA_types, dtype=tf.int32)
    
    # Target tensor: shape (batch_size, 910), float32 values in [0,1]
    out_dim = 910
    targ = tf.random.uniform(shape=(batch_size, out_dim), minval=0.0, maxval=1.0, dtype=tf.float32)
    
    # Model call expects only samples input; to be compatible with loss, return tuple
    # Here, since MyModel call only takes samples, and loss takes targ too,
    # we can return (samples, targ) for downstream usage (train_step-like)
    return samples, targ

