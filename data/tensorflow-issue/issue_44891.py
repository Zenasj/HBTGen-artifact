# tf.random.uniform((1, 256, 256, 3), dtype=tf.float32) ← Input batch assumed batch=1 here

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, Activation, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding.upper()
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        # Use tf.compat.v1.variable_scope for TF 2.x compatibility with variable scope.
        with tf.compat.v1.variable_scope(self.name):
            mask = tf.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')

            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3]
                )
            # Flatten mask and updates for scatter_nd
            flat_mask = tf.expand_dims(tf.reshape(mask, [-1]), axis=1)
            flat_updates = tf.reshape(updates, [-1])

            # Total number of output elements = product of output_shape
            total_output_elements = tf.reduce_prod(output_shape)

            ret = tf.scatter_nd(flat_mask, flat_updates, [total_output_elements])

            # Get static shape info if possible for reshape
            updates_shape_static = updates.shape
            out_shape = [
                -1,
                updates_shape_static[1] * self.size[0] if updates_shape_static[1] is not None else None,
                updates_shape_static[2] * self.size[1] if updates_shape_static[2] is not None else None,
                updates_shape_static[3] if updates_shape_static[3] is not None else None
            ]
            return tf.reshape(ret, out_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )


class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3), n_labels=1, kernel=3, pool_size=(2,2), output_mode="softmax"):
        super(MyModel, self).__init__()
        self.input_shape_ = input_shape
        self.n_labels = n_labels
        self.kernel = kernel
        self.pool_size = pool_size
        self.output_mode = output_mode

        # Encoder layers
        self.conv1_1 = Conv2D(64, (kernel, kernel), padding="same")
        self.bn1_1 = BatchNormalization()
        self.act1_1 = Activation("relu")
        self.conv1_2 = Conv2D(64, (kernel, kernel), padding="same")
        self.bn1_2 = BatchNormalization()
        self.act1_2 = Activation("relu")
        self.pool1 = MaxPoolingWithArgmax2D(pool_size)

        self.conv2_1 = Conv2D(128, (kernel, kernel), padding="same")
        self.bn2_1 = BatchNormalization()
        self.act2_1 = Activation("relu")
        self.conv2_2 = Conv2D(128, (kernel, kernel), padding="same")
        self.bn2_2 = BatchNormalization()
        self.act2_2 = Activation("relu")
        self.pool2 = MaxPoolingWithArgmax2D(pool_size)

        self.conv3_1 = Conv2D(256, (kernel, kernel), padding="same")
        self.bn3_1 = BatchNormalization()
        self.act3_1 = Activation("relu")
        self.conv3_2 = Conv2D(256, (kernel, kernel), padding="same")
        self.bn3_2 = BatchNormalization()
        self.act3_2 = Activation("relu")
        self.conv3_3 = Conv2D(256, (kernel, kernel), padding="same")
        self.bn3_3 = BatchNormalization()
        self.act3_3 = Activation("relu")
        self.pool3 = MaxPoolingWithArgmax2D(pool_size)

        self.conv4_1 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn4_1 = BatchNormalization()
        self.act4_1 = Activation("relu")
        self.conv4_2 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn4_2 = BatchNormalization()
        self.act4_2 = Activation("relu")
        self.conv4_3 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn4_3 = BatchNormalization()
        self.act4_3 = Activation("relu")
        self.pool4 = MaxPoolingWithArgmax2D(pool_size)

        self.conv5_1 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn5_1 = BatchNormalization()
        self.act5_1 = Activation("relu")
        self.conv5_2 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn5_2 = BatchNormalization()
        self.act5_2 = Activation("relu")
        self.conv5_3 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn5_3 = BatchNormalization()
        self.act5_3 = Activation("relu")
        self.pool5 = MaxPoolingWithArgmax2D(pool_size)

        # Decoder layers

        self.unpool1 = MaxUnpooling2D(pool_size)
        self.conv_d14 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn_d14 = BatchNormalization()
        self.act_d14 = Activation("relu")
        self.conv_d15 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn_d15 = BatchNormalization()
        self.act_d15 = Activation("relu")
        self.conv_d16 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn_d16 = BatchNormalization()
        self.act_d16 = Activation("relu")

        self.unpool2 = MaxUnpooling2D(pool_size)
        self.conv_d17 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn_d17 = BatchNormalization()
        self.act_d17 = Activation("relu")
        self.conv_d18 = Conv2D(512, (kernel, kernel), padding="same")
        self.bn_d18 = BatchNormalization()
        self.act_d18 = Activation("relu")
        self.conv_d19 = Conv2D(256, (kernel, kernel), padding="same")
        self.bn_d19 = BatchNormalization()
        self.act_d19 = Activation("relu")

        self.unpool3 = MaxUnpooling2D(pool_size)
        self.conv_d20 = Conv2D(256, (kernel, kernel), padding="same")
        self.bn_d20 = BatchNormalization()
        self.act_d20 = Activation("relu")
        self.conv_d21 = Conv2D(256, (kernel, kernel), padding="same")
        self.bn_d21 = BatchNormalization()
        self.act_d21 = Activation("relu")
        self.conv_d22 = Conv2D(128, (kernel, kernel), padding="same")
        self.bn_d22 = BatchNormalization()
        self.act_d22 = Activation("relu")

        self.unpool4 = MaxUnpooling2D(pool_size)
        self.conv_d23 = Conv2D(128, (kernel, kernel), padding="same")
        self.bn_d23 = BatchNormalization()
        self.act_d23 = Activation("relu")
        self.conv_d24 = Conv2D(64, (kernel, kernel), padding="same")
        self.bn_d24 = BatchNormalization()
        self.act_d24 = Activation("relu")

        self.unpool5 = MaxUnpooling2D(pool_size)
        self.conv_d25 = Conv2D(64, (kernel, kernel), padding="same")
        self.bn_d25 = BatchNormalization()
        self.act_d25 = Activation("relu")

        self.conv_d26 = Conv2D(n_labels, (1, 1), padding="valid")
        self.bn_d26 = BatchNormalization()
        # final reshape and activation will be done in call()

    def call(self, inputs, training=None):
        # Encoder
        conv_1 = self.conv1_1(inputs)
        conv_1 = self.bn1_1(conv_1, training=training)
        conv_1 = self.act1_1(conv_1)
        conv_2 = self.conv1_2(conv_1)
        conv_2 = self.bn1_2(conv_2, training=training)
        conv_2 = self.act1_2(conv_2)
        pool_1, mask_1 = self.pool1(conv_2)

        conv_3 = self.conv2_1(pool_1)
        conv_3 = self.bn2_1(conv_3, training=training)
        conv_3 = self.act2_1(conv_3)
        conv_4 = self.conv2_2(conv_3)
        conv_4 = self.bn2_2(conv_4, training=training)
        conv_4 = self.act2_2(conv_4)
        pool_2, mask_2 = self.pool2(conv_4)

        conv_5 = self.conv3_1(pool_2)
        conv_5 = self.bn3_1(conv_5, training=training)
        conv_5 = self.act3_1(conv_5)
        conv_6 = self.conv3_2(conv_5)
        conv_6 = self.bn3_2(conv_6, training=training)
        conv_6 = self.act3_2(conv_6)
        conv_7 = self.conv3_3(conv_6)
        conv_7 = self.bn3_3(conv_7, training=training)
        conv_7 = self.act3_3(conv_7)
        pool_3, mask_3 = self.pool3(conv_7)

        conv_8 = self.conv4_1(pool_3)
        conv_8 = self.bn4_1(conv_8, training=training)
        conv_8 = self.act4_1(conv_8)
        conv_9 = self.conv4_2(conv_8)
        conv_9 = self.bn4_2(conv_9, training=training)
        conv_9 = self.act4_2(conv_9)
        conv_10 = self.conv4_3(conv_9)
        conv_10 = self.bn4_3(conv_10, training=training)
        conv_10 = self.act4_3(conv_10)
        pool_4, mask_4 = self.pool4(conv_10)

        conv_11 = self.conv5_1(pool_4)
        conv_11 = self.bn5_1(conv_11, training=training)
        conv_11 = self.act5_1(conv_11)
        conv_12 = self.conv5_2(conv_11)
        conv_12 = self.bn5_2(conv_12, training=training)
        conv_12 = self.act5_2(conv_12)
        conv_13 = self.conv5_3(conv_12)
        conv_13 = self.bn5_3(conv_13, training=training)
        conv_13 = self.act5_3(conv_13)
        pool_5, mask_5 = self.pool5(conv_13)

        # Decoder

        unpool_1 = self.unpool1([pool_5, mask_5])

        conv_14 = self.conv_d14(unpool_1)
        conv_14 = self.bn_d14(conv_14, training=training)
        conv_14 = self.act_d14(conv_14)
        conv_15 = self.conv_d15(conv_14)
        conv_15 = self.bn_d15(conv_15, training=training)
        conv_15 = self.act_d15(conv_15)
        conv_16 = self.conv_d16(conv_15)
        conv_16 = self.bn_d16(conv_16, training=training)
        conv_16 = self.act_d16(conv_16)

        unpool_2 = self.unpool2([conv_16, mask_4])

        conv_17 = self.conv_d17(unpool_2)
        conv_17 = self.bn_d17(conv_17, training=training)
        conv_17 = self.act_d17(conv_17)
        conv_18 = self.conv_d18(conv_17)
        conv_18 = self.bn_d18(conv_18, training=training)
        conv_18 = self.act_d18(conv_18)
        conv_19 = self.conv_d19(conv_18)
        conv_19 = self.bn_d19(conv_19, training=training)
        conv_19 = self.act_d19(conv_19)

        unpool_3 = self.unpool3([conv_19, mask_3])

        conv_20 = self.conv_d20(unpool_3)
        conv_20 = self.bn_d20(conv_20, training=training)
        conv_20 = self.act_d20(conv_20)
        conv_21 = self.conv_d21(conv_20)
        conv_21 = self.bn_d21(conv_21, training=training)
        conv_21 = self.act_d21(conv_21)
        conv_22 = self.conv_d22(conv_21)
        conv_22 = self.bn_d22(conv_22, training=training)
        conv_22 = self.act_d22(conv_22)

        unpool_4 = self.unpool4([conv_22, mask_2])

        conv_23 = self.conv_d23(unpool_4)
        conv_23 = self.bn_d23(conv_23, training=training)
        conv_23 = self.act_d23(conv_23)
        conv_24 = self.conv_d24(conv_23)
        conv_24 = self.bn_d24(conv_24, training=training)
        conv_24 = self.act_d24(conv_24)

        unpool_5 = self.unpool5([conv_24, mask_1])

        conv_25 = self.conv_d25(unpool_5)
        conv_25 = self.bn_d25(conv_25, training=training)
        conv_25 = self.act_d25(conv_25)

        conv_26 = self.conv_d26(conv_25)
        conv_26 = self.bn_d26(conv_26, training=training)

        # Reshape output to (H*W, n_labels)
        # Note: input_shape[0] = height, input_shape[1] = width
        output_reshape = Reshape(
            (self.input_shape_[0] * self.input_shape_[1], self.n_labels)
        )(conv_26)

        outputs = Activation(self.output_mode)(output_reshape)

        return outputs


def my_model_function():
    # Instantiate model with default parameters matching original segnet call in issue
    return MyModel(input_shape=(256, 256, 3), n_labels=1)


def GetInput():
    # Return a single random input tensor with shape (1, 256, 256, 3)
    # Batch size 1 as typical for segmentation networks during inference or training
    return tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)

