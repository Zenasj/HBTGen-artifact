# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← input shape inferred from example Input(shape=(32, 32, 3))

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, concatenate
from tensorflow.keras import initializers
import tensorflow.keras.backend as K

def _conv_layer(filters, kernel_size, strides=(1, 1), padding='same', name=None):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='he_normal', name=name)

def _normalize_depth_vars(depth_k, depth_v, filters):
    # depth_k and depth_v can be float (fraction of filters) or int (absolute)
    if isinstance(depth_k, float):
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if isinstance(depth_v, float):
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v

class AttentionAugmentation2D(Layer):
    def __init__(self, depth_k, depth_v, num_heads, relative=True, **kwargs):
        super(AttentionAugmentation2D, self).__init__(**kwargs)
        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (depth_k, num_heads))
        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (depth_v, num_heads))
        if depth_k // num_heads < 1:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! Given depth_k = %d, num_heads = %d' % (depth_k, num_heads))
        if depth_v // num_heads < 1:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! Given depth_v = %d, num_heads = %d' % (depth_v, num_heads))

        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative = relative

        self.axis = 1 if K.image_data_format() == 'channels_first' else -1

    def build(self, input_shape):
        self._shape = input_shape

        # Normalize depth_k, depth_v to int values (input_shape - filters)
        filters = input_shape[self.axis]
        self.depth_k, self.depth_v = _normalize_depth_vars(self.depth_k, self.depth_v, filters)

        if self.axis == 1:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape

        # Save spatial dims
        self._height = height
        self._width = width

        if self.relative:
            dk_per_head = self.depth_k // self.num_heads
            self.key_relative_w = self.add_weight(
                'key_rel_w',
                shape=[2 * self._width - 1, dk_per_head],
                initializer=initializers.RandomNormal(stddev=dk_per_head ** -0.5)
            )
            self.key_relative_h = self.add_weight(
                'key_rel_h',
                shape=[2 * self._height - 1, dk_per_head],
                initializer=initializers.RandomNormal(stddev=dk_per_head ** -0.5)
            )
        else:
            self.key_relative_w = None
            self.key_relative_h = None

        super(AttentionAugmentation2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.axis == 1:
            # channels_first -> channels_last for computations
            inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])

        # Split q, k, v from channels
        q, k, v = tf.split(inputs, [self.depth_k, self.depth_k, self.depth_v], axis=-1)

        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)

        depth_k_heads = self.depth_k // self.num_heads
        q = tf.cast(q, tf.float32)  # Ensure float32 for precision/stability
        k = tf.cast(k, tf.float32)
        v = tf.cast(v, tf.float32)

        q *= depth_k_heads ** -0.5  # scale the query

        # Shape = [batch, num_heads, height*width, depth_k_heads]
        flat_q = tf.reshape(q, [-1, self.num_heads, self._height * self._width, depth_k_heads])
        flat_k = tf.reshape(k, [-1, self.num_heads, self._height * self._width, depth_k_heads])
        flat_v = tf.reshape(v, [-1, self.num_heads, self._height * self._width, self.depth_v // self.num_heads])

        # Compute dot product attention logits
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)  # [B, heads, HW, HW]

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = tf.nn.softmax(logits, axis=-1)  # attention weights
        attn_out = tf.matmul(weights, flat_v)     # weighted sum of values

        # Reshape attn_out to [batch, num_heads, height, width, depth_v_per_head]
        attn_out = tf.reshape(attn_out,
                              [-1, self.num_heads, self._height, self._width, self.depth_v // self.num_heads])

        attn_out = self.combine_heads_2d(attn_out)  # combine heads into channels

        if self.axis == 1:
            # back to channels first format
            attn_out = K.permute_dimensions(attn_out, [0, 3, 1, 2])

        attn_out.set_shape(self.compute_output_shape(self._shape))
        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)

    def split_heads_2d(self, x):
        # x shape: [batch, height, width, channels]
        # split channels into [num_heads, channels_per_head]
        shape = tf.shape(x)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        x = tf.reshape(x, [batch, height, width, self.num_heads, channels // self.num_heads])
        x = tf.transpose(x, [0, 3, 1, 2, 4])  # [batch, num_heads, height, width, channels_per_head]
        return x

    def combine_heads_2d(self, x):
        # inverse of split_heads_2d
        # x shape: [batch, num_heads, height, width, channels_per_head]
        x = tf.transpose(x, [0, 2, 3, 1, 4])  # [batch, height, width, num_heads, channels_per_head]
        shape = tf.shape(x)
        batch, height, width, num_heads, channels_per_head = shape[0], shape[1], shape[2], shape[3], shape[4]
        return tf.reshape(x, [batch, height, width, num_heads * channels_per_head])

    def relative_logits(self, q):
        # q shape: [batch, num_heads, height, width, depth_per_head]
        h_rel_logits = self.relative_logits_1d(q, self.key_relative_h, self._height, self._width,
                                              transpose_mask=[0,1,4,2,5,3])
        # permute q for width dimension
        q_w = tf.transpose(q, [0,1,3,2,4])
        w_rel_logits = self.relative_logits_1d(q_w, self.key_relative_w, self._width, self._height,
                                              transpose_mask=[0,1,4,2,5,3])
        return h_rel_logits, w_rel_logits

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        # q: [B, heads, H, W, depth]
        # rel_k: [2*L - 1, depth]
        # Step 1: compute einsum b h x y d, m d -> b h x y m
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)

        # Reshape & convert relative to absolute indexing
        B = tf.shape(rel_logits)[0]
        Nh = tf.shape(rel_logits)[1]
        L = H

        rel_logits = tf.reshape(rel_logits, [B, Nh * L, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = tf.reshape(rel_logits, [B, Nh, L, W, W])

        # Expand dims and tile
        rel_logits = tf.expand_dims(rel_logits, 3)  # [B, Nh, L, 1, W, W]
        rel_logits = tf.tile(rel_logits, [1, 1, 1, L, 1, 1])  # [B, Nh, L, L, W, W]
        rel_logits = tf.transpose(rel_logits, perm=transpose_mask)
        rel_logits = tf.reshape(rel_logits, [B, Nh, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        # Converts relative indexing to absolute indexing.
        # x shape: [B, Nh, L, 2L - 1]
        B, Nh, L = tf.unstack(tf.shape(x))
        # Pad last dimension
        col_pad = tf.zeros([B, Nh, L, 1], dtype=x.dtype)
        x = tf.concat([x, col_pad], axis=3)  # [B, Nh, L, 2L]

        # Flatten and pad
        flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = tf.zeros([B, Nh, L - 1], dtype=x.dtype)
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)

        final_x = tf.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def get_config(self):
        config = {
            'depth_k': self.depth_k,
            'depth_v': self.depth_v,
            'num_heads': self.num_heads,
            'relative': self.relative,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Constants from example: filters=20, kernel_size=3x3, depth_k=0.2, depth_v=0.2, num_heads=4
        self.filters = 20
        self.kernel_size = (3, 3)
        self.depth_k_frac = 0.2
        self.depth_v_frac = 0.2
        self.num_heads = 4
        self.relative_encodings = True

        self.channel_axis = -1 if K.image_data_format() == 'channels_last' else 1

        # Normalize depth_k and depth_v to integer based on filters
        depth_k, depth_v = _normalize_depth_vars(self.depth_k_frac, self.depth_v_frac, self.filters)
        self.depth_k = depth_k
        self.depth_v = depth_v

        # Define layers:
        self.conv_out_layer = _conv_layer(self.filters - self.depth_v,
                                          self.kernel_size,
                                          strides=(1, 1),
                                          padding='same')

        # qkv conv layer producing 2*depth_k + depth_v channels
        self.qkv_conv_layer = _conv_layer(2 * self.depth_k + self.depth_v,
                                          kernel_size=(1, 1),
                                          strides=(1, 1),
                                          padding='same')

        self.attention_aug_layer = AttentionAugmentation2D(depth_k, depth_v, self.num_heads, self.relative_encodings)
        self.attn_out_proj = _conv_layer(self.depth_v, kernel_size=(1, 1))

        self.batchnorm = BatchNormalization()

    def call(self, inputs, training=False):
        conv_out = self.conv_out_layer(inputs)  # Regular conv branch

        qkv = self.qkv_conv_layer(inputs)  # qkv conv branch

        attn_out = self.attention_aug_layer(qkv)  # Attention augmentation
        attn_out = self.attn_out_proj(attn_out)

        output = concatenate([conv_out, attn_out], axis=self.channel_axis)
        output = self.batchnorm(output, training=training)
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return random input tensor matching input shape: batch=1, height=32, width=32, channels=3
    # Use float32 dtype as per TensorFlow standard for image data
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)

