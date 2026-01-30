import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

class RetinaNet(tf.keras.Model):
    def __init__(self, config):
        super(RetinaNet, self).__init__()
        self.config = config
        self.num_classes = config['num_classes']
        self.weight_decay = config['weight_decay']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.lr = config['lr']

        image = tf.keras.Input(shape=[None, None, 3], batch_size=self.batch_size, dtype=tf.float32)
        self.backone = backone.model[config['backone']](
            image, config['backone_conv_trainable'], config['backone_bn_trainable'],
            weight=backone.weight[config['backone']]
        )
        self.opt = tf.keras.optimizers.SGD(self.lr, momentum=0.9)

        if config['mode'] == 'train':
            gt = tf.keras.Input(shape=[None, 5], batch_size=self.batch_size, dtype=tf.float32)
            self.model = self._build_graph(image, gt)
        else:
            self.model = self._build_graph(image)

    def _build_graph(self, image, gt=None):
        num_fpn_layers = 5
        fpn_channels = 256
        endpoints = self.backone(image)
        fpn = fpn_generator(endpoints[1:], fpn_channels, num_fpn_layers, mode='dconv')
        p3, p4, p5, p6, p7 = fpn
        dw_rate = [8., 16., 32., 64., 128.]
        anchors = [
            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],
        ]
        anchors_all = anchor_generator(
            fpn, anchors, dw_rate, flatten=True
        )
        anchors_all = tf.concat(anchors_all, axis=0)
        self.cla_head = self._cla_head(fpn_channels, 5)
        self.reg_head = self._reg_head(fpn_channels, 5)
        p3c = self.cla_head(p3)
        p3r = self.reg_head(p3)
        p4c = self.cla_head(p4)
        p4r = self.reg_head(p4)
        p5c = self.cla_head(p5)
        p5r = self.reg_head(p5)
        p6c = self.cla_head(p6)
        p6r = self.reg_head(p6)
        p7c = self.cla_head(p7)
        p7r = self.reg_head(p7)
        pc = tf.concat([p3c, p4c, p5c, p6c, p7c], axis=1)
        pr = tf.concat([p3r, p4r, p5r, p6r, p7r], axis=1)
        pc = tf.nn.sigmoid(pc)
        if self.mode == 'train':
            i = 0
            loss = tf.constant([0., 0.], dtype=tf.float32, shape=[1, 2])
            cond = lambda loss, i: tf.less(i, self.batch_size)
            body = lambda loss, i: (
                tf.add(loss, self._compute_one_image_loss(
                    tf.gather(gt, i),
                    anchors_all,
                    tf.gather(pc, i),
                    tf.gather(pr, i))
                       ),
                tf.add(i, 1)
            )
            loss, _ = tf.while_loop(cond, body, (loss, i))
            loss /= self.batch_size
            return tf.keras.Model(inputs=[image, gt], outputs=loss, name='retinanet')
        else:
            nms_score_threshold = 0.5
            nms_max_boxes = 100
            nms_iou_threshold = 0.45
            pr = pr[0, ...]
            confidence = pc[0, ...]
            y1x1y2x2 = bbox_decode(anchors_all, pr, normlization=[10., 10., 5., 5.])
            filter_mask = tf.greater_equal(confidence, nms_score_threshold)
            scores = []
            class_id = []
            bbox = []
            for i in range(self.num_classes):
                scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                bboxi = tf.boolean_mask(y1x1y2x2, filter_mask[:, i])
                selected_indices = tf.image.non_max_suppression(
                    bboxi, scoresi, nms_max_boxes, nms_iou_threshold,
                )
                scores.append(tf.gather(scoresi, selected_indices))
                bbox.append(tf.gather(bboxi, selected_indices))
                class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices), tf.int32) * i)
            bbox = tf.concat(bbox, axis=0)
            scores = tf.concat(scores, axis=0)
            class_id = tf.concat(class_id, axis=0) + 1
            detection_pred = [scores, bbox, class_id]
            return tf.keras.Model(inputs=image, outputs=detection_pred, name='retinanet')

    def _compute_one_image_loss(self, gt, anchors, pc, pr):
        slice_index = tf.argmin(gt, axis=0)[0]
        gt = tf.gather(gt, tf.range(0, slice_index, dtype=tf.int64))
        gt_bbox = gt[:, 0:4]
        label = tf.cast(gt[..., 4:5], dtype=tf.int32) - 1
        pos_threshold = 0.5
        neg_threshold = 0.4
        gaiou = bbox_iou(gt_bbox, anchors)
        pos_pc, pos_label, pos_pr, pos_gt_bbox, pos_a, neg_pc = partition_pos_neg_samples(
            gt_bbox, label, gaiou, pc, pr, anchors, pos_threshold, neg_threshold
        )
        pos_gr = bbox_encode(pos_gt_bbox, pos_a, normlization=[10., 10, 5., 5.])
        reg_loss = tf.reduce_sum(smooth_l1_loss(pos_pr-pos_gr))
        pos_label = tf.one_hot(pos_label, self.num_classes, dtype=tf.float32)
        cla_loss = focal_loss(pos_pc, pos_label, neg_pc, alpha=0.25, gamma=2.)
        reg_loss = tf.reshape(reg_loss, [1, 1])
        cla_loss = tf.reshape(cla_loss, [1, 1])
        loss = tf.concat([cla_loss, reg_loss], axis=-1)
        return loss

    def _cla_head(self, input_channels, anchors):
        x = tf.keras.Input(shape=[None, None, input_channels], dtype=tf.float32)
        conv = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal')(x)
        bn = layers.BatchNormalization(3, epsilon=1.001e-5)(conv)
        relu = layers.Activation('relu')(bn)
        conv = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal')(relu)
        bn = layers.BatchNormalization(3, epsilon=1.001e-5)(conv)
        relu = layers.Activation('relu')(bn)
        conv = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal')(relu)
        bn = layers.BatchNormalization(3, epsilon=1.001e-5)(conv)
        relu = layers.Activation('relu')(bn)
        pred = layers.Conv2D(self.num_classes * anchors, 3, 1, 'same', kernel_initializer='he_normal',
                             bias_initializer=tf.constant_initializer(-4.595))(relu)
        pred = tf.reshape(pred, [self.batch_size, -1, self.num_classes])
        return tf.keras.Model(inputs=x, outputs=pred, name='cla_head')

    def _reg_head(self, input_channels, anchors):
        x = tf.keras.Input(shape=[None, None, input_channels], dtype=tf.float32)
        conv = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal')(x)
        bn = layers.BatchNormalization(3, epsilon=1.001e-5)(conv)
        relu = layers.Activation('relu')(bn)
        conv = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal')(relu)
        bn = layers.BatchNormalization(3, epsilon=1.001e-5)(conv)
        relu = layers.Activation('relu')(bn)
        conv = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal')(relu)
        bn = layers.BatchNormalization(3, epsilon=1.001e-5)(conv)
        relu = layers.Activation('relu')(bn)
        pred = layers.Conv2D(4 * anchors, 3, 1, 'same', kernel_initializer='he_normal')(relu)
        pred = tf.reshape(pred, [self.batch_size, -1, 4])
        return tf.keras.Model(inputs=x, outputs=pred, name='reg_head')

    def save_weights(self, filepath, overwrite=True, save_format=None):
        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name)

config = {
    'num_classes': 20,
    'batch_size':batch_size,
    'mode': 'train',
    'lr': lr,
    'weight_decay':1e-4,
    'backone': 'resnetv1_18',
    'backone_conv_trainable': True,
    'backone_bn_trainable': True,
}
ssd = RetinaNet(config)
ssd.save_weights('saved_weights/1.tf')

config = {
    'num_classes': 20,
    'batch_size':batch_size,
    'mode': 'test',
    'lr': lr,
    'weight_decay':1e-4,
    'backone': 'resnetv1_18',
    'backone_conv_trainable': True,
    'backone_bn_trainable': True,
}
ssd = RetinaNet(config)
ssd.load_weights('saved_weights/1.tf')

def _head(self, input_channels, anchors):
    x = tf.keras.Input(shape=[None, None, input_channels], dtype=tf.float32)
    conv1 = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal', name='cla_conv1')(x)
    bn1 = layers.BatchNormalization(3, epsilon=1.001e-5, name='cla_bn1')(conv1)
    relu1 = layers.Activation('relu', name='cla_relu1')(bn1)
    conv1 = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal', name='cla_conv2')(relu1)
    bn1 = layers.BatchNormalization(3, epsilon=1.001e-5, name='cla_bn2')(conv1)
    relu1 = layers.Activation('relu', name='cla_relu2')(bn1)
    conv1 = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal', name='cla_conv3')(relu1)
    bn1 = layers.BatchNormalization(3, epsilon=1.001e-5, name='cla_bn3')(conv1)
    relu1 = layers.Activation('relu', name='cla_relu3')(bn1)
    cla_pred = layers.Conv2D(self.num_classes * anchors, 3, 1, 'same', kernel_initializer='he_normal',
                          name='cla_conv4', bias_initializer=tf.constant_initializer(-4.595))(relu1)
    cla_pred = tf.reshape(cla_pred, [self.batch_size, -1, self.num_classes])

    conv2 = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal', name='reg_conv1')(x)
    bn2 = layers.BatchNormalization(3, epsilon=1.001e-5, name='reg_bn1')(conv2)
    relu2 = layers.Activation('relu', name='reg_relu1')(bn2)
    conv2 = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal', name='reg_conv2')(relu2)
    bn2 = layers.BatchNormalization(3, epsilon=1.001e-5, name='reg_bn2')(conv2)
    relu2 = layers.Activation('relu', name='reg_relu2')(bn2)
    conv2 = layers.Conv2D(256, 3, 1, 'same', kernel_initializer='he_normal', name='reg_conv3')(relu2)
    bn2 = layers.BatchNormalization(3, epsilon=1.001e-5, name='reg_bn3')(conv2)
    relu2 = layers.Activation('relu', name='reg_relu3')(bn2)
    reg_pred = layers.Conv2D(4 * anchors, 3, 1, 'same', kernel_initializer='he_normal', name='reg_conv4')(relu2)
    reg_pred = tf.reshape(reg_pred, [self.batch_size, -1, 4])
    return tf.keras.Model(inputs=x, outputs=[cla_pred, reg_pred])