import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPool2D, ZeroPadding2D
from tensorflow.keras import Model, Sequential


feature_extractor_config = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

feature_pyramid_config = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}

head_config = {
    '300': [4,6,6,6,4,4]
}


def create_feature_extractor(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2D(pool_size=2, strides=2)]
        elif v == 'C':
            layers += [ZeroPadding2D(padding=((0,1), (0,1))), MaxPool2D(pool_size=2, strides=2)]
        else:
            pad = ZeroPadding2D(1)
            conv2d = Conv2D(v, kernel_size=3)
            if batch_norm:
                layers += [pad, conv2d, BatchNormalization(v), ReLU()]
            else:
                layers += [pad, conv2d, ReLU()]
            in_channels = v
    pad5 = ZeroPadding2D(padding=1)
    pool5 = MaxPool2D(pool_size=3, strides=1)
    pad6 = ZeroPadding2D(padding=6)
    conv6 = Conv2D(1024, kernel_size=3, dilation_rate=6)
    conv7 = Conv2D(1024, kernel_size=1)
    layers += [pad5, pool5, pad6, conv6,
               ReLU(), conv7, ReLU()]
    return layers

def create_feature_pyramid(cfg, size=300):
    layers = []
    # in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if k==0 or in_channels != 'S':
            if v == 'S':
                layers += [ZeroPadding2D(padding=1), Conv2D(cfg[k + 1], kernel_size=(1, 3)[flag], strides=2)]
            else:
                layers += [Conv2D(v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(Conv2D(128, kernel_size=1, strides=1))
        layers.append(ZeroPadding2D(padding=1))
        layers.append(Conv2D(256, kernel_size=4, strides=1))
    return layers

def create_head(cfg, num_classes):
    reg_layers  = []
    cls_layers = []
    for num_bboxes in cfg:
        cls_layers.append(Sequential([ZeroPadding2D(padding=1), Conv2D(num_bboxes * num_classes, kernel_size=3)]))
        reg_layers.append(Sequential([ZeroPadding2D(padding=1), Conv2D(num_bboxes * 4, kernel_size=3)]))
    head = {'reg': reg_layers, 'cls': cls_layers}

    return head

class L2Norm(Layer):
    def __init__(self, in_channels, scale):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.w = self.add_weight(
            name='w',
            shape=(self.in_channels,),
            initializer=tf.constant_initializer(20),
            trainable=True,
            dtype=self.dtype
        )

    def call(self, x):
        norm = tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=3, keepdims=True)) + self.eps
        x = tf.truediv(x, norm)
        out = self.w * x

        return out

class VGG(Model):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature_extractor = create_feature_extractor(feature_extractor_config['300'])
        self.l2_norm = L2Norm(512, scale=20)
        self.feature_pyramid = create_feature_pyramid(feature_pyramid_config['300'], size=300)
        self.num_classes = 21

        self.head = create_head(head_config['300'], self.num_classes)


    def call(self, x):
        n, h, w, c = x.shape
        features = []
        for i in range(34):
            x = self.feature_extractor[i](x)
        s = self.l2_norm(x)

        features.append(s)

        for i in range(34, len(self.feature_extractor)):
            x = self.feature_extractor[i](x)
        features.append(x)

        for k, v in enumerate(self.feature_pyramid):
            x = tf.nn.relu(v(x))
            if k in [2,5,7,9]:
                features.append(x)

        regressions = []
        classifications = []

        for k, v in enumerate(features):
            regressions.append(tf.reshape(self.head['reg'][k](v), [32, -1, 4]))
            classifications.append(tf.reshape(self.head['cls'][k](v), [32, -1, self.num_classes]))

        regressions = tf.concat(regressions, axis=1)
        classifications = tf.concat(classifications, axis=1)

        return [regressions, classifications]
model = VGG()

optimizer = tf.keras.optimizers.Adam()


smooth_l1_loss = tf.keras.losses.Huber(
    delta=1.0, reduction=tf.keras.losses.Reduction.NONE
)

ce_loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE
)

@tf.function
def train_step(input_imgs, target_bboxes, target_labels):
    outputs = model(input_imgs)

    regressions = outputs[0]
    classifications = outputs[1]

    pos_mask = target_labels > 0
    neg_mask = target_labels == 0
    num_pos = tf.reduce_sum(tf.cast(pos_mask, tf.float32))

    predicted_bboxes = regressions[pos_mask]
    target_bboxes = target_bboxes[pos_mask]

    bbox_loss = tf.reduce_sum(smooth_l1_loss(target_bboxes, predicted_bboxes))

    cls_loss = ce_loss(tf.one_hot(tf.cast(target_labels, tf.int32), classifications.shape[2]), classifications)

    pos_cls_loss = cls_loss[pos_mask]
    neg_cls_loss = cls_loss[neg_mask]
    neg_cls_loss = tf.sort(neg_cls_loss, direction="DESCENDING")
    neg_cls_loss = neg_cls_loss[: 3 * tf.cast(num_pos, tf.int32)]

    cls_loss = tf.reduce_sum(
        tf.reduce_sum(pos_cls_loss) + tf.reduce_sum(neg_cls_loss)
    )
    loss = bbox_loss + cls_loss
    loss = loss / num_pos

    gradients = tf.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

while True:
    imgs = tf.random.uniform((32, 300, 300, 3), minval=0, maxval=255)
    img_bboxes = tf.random.uniform((32, 8732, 4), minval=0, maxval=1)
    img_labels = tf.random.uniform((32, 8732), minval=0, maxval=21, dtype=tf.int32)
    train_step(imgs, img_bboxes, img_labels)