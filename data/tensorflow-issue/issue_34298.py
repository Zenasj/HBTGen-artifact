# tf.random.uniform((batch_size, None, None, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import layers

# Note: The original issue references a backbone module (backone) and fpn_generator, anchor_generator, bbox_decode,
# bbox_iou, partition_pos_neg_samples, bbox_encode, smooth_l1_loss, focal_loss which are not defined here.
# For the purpose of creating a self-contained model class, I will replace those with placeholders or simple
# pass-through constructs where appropriate with comments indicating assumptions.
#
# The inputs and outputs, shapes, and head structure are based on provided code.
# The fused _head model is used for both classification and regression heads to avoid the variable loading issue.
# The mode switch controls whether loss or detection outputs are returned similar to original.
#
# Assumptions:
# - Input shape: (batch_size, height, width, 3) with batch_size known at init.
# - The classification head output shape is (batch_size, n_anchors, num_classes)
# - The regression head output shape is (batch_size, n_anchors, 4)
# - For demonstration, dummy implementations replace missing components.
# - The forward returns either loss tensor or detection outputs, wrapped in a keras.Model.
# - Load and save weights methods delegate to inner model for compatibility.
# - The backbone and other submodules are placeholders.
# - For GetInput(), a random image tensor of shape (batch_size, 256, 256, 3) is generated,
#   assuming a reasonable input size for RetinaNet.

# Placeholder implementations for missing components to allow code to run:
def dummy_backbone_model(image, conv_trainable, bn_trainable, weight=None):
    # Returns dummy feature maps simulating endpoints
    # Example: list of tensors with shapes increasing in stride (height//4, width//4, channels), etc.
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(image)
    p3 = layers.Conv2D(256, 3, padding='same', activation='relu')(x)  # Simulate p3
    p4 = layers.Conv2D(256, 3, padding='same', strides=2, activation='relu')(p3)
    p5 = layers.Conv2D(256, 3, padding='same', strides=2, activation='relu')(p4)
    p6 = layers.Conv2D(256, 3, padding='same', strides=2, activation='relu')(p5)
    p7 = layers.Conv2D(256, 3, padding='same', strides=2, activation='relu')(p6)
    # Return dummy list with endpoints (simulate backbone stages)
    # According to original code: starts from endpoint[1:], so first can be dummy
    return [x, p3, p4, p5, p6, p7]


def fpn_generator(features, fpn_channels, num_layers, mode='dconv'):
    # Dummy feature pyramid network generator returning 5 feature maps
    # Just return the features themselves truncated or adjusted
    # For simplicity, return the last 5 passed feature maps directly
    # Shape checks and joining operations omitted for brevity
    return features[:num_layers]


def anchor_generator(fpn_features, anchors, dw_rate, flatten=True):
    # Dummy anchor generator returns list of tensors with anchor shapes
    # For simplicity, just create tensors matching spatial dims and anchor counts
    ret = []
    for i, f in enumerate(fpn_features):
        shape = tf.shape(f)
        # Calculate number of anchors per location: len(anchors[i])
        num_anchors = len(anchors[i])
        h, w = shape[1], shape[2]
        # create dummy anchors tensor for each spatial location * anchors
        anchors_tensor = tf.zeros((h * w * num_anchors, 4), dtype=tf.float32)
        ret.append(anchors_tensor)
    return ret


def bbox_decode(anchors_all, pr, normlization):
    # Dummy bbox decode returns anchors_all + pr (fake decode)
    return anchors_all + pr


def bbox_iou(gt_bbox, anchors):
    # Dummy bbox iou returns random floats as IoU values
    return tf.random.uniform((tf.shape(anchors)[0],), minval=0, maxval=1)


def partition_pos_neg_samples(gt_bbox, label, gaiou, pc, pr, anchors, pos_thr, neg_thr):
    # Random partition returning dummy tensors consistent with shapes expected
    # For simplicity, just return input with slices or dummy tensors
    pos_pc = tf.constant(0.9, shape=(1,1), dtype=tf.float32)
    pos_label = tf.constant(0, shape=(1,), dtype=tf.int32)
    pos_pr = tf.constant(0.1, shape=(1,4), dtype=tf.float32)
    pos_gt_bbox = tf.constant(0.0, shape=(1,4), dtype=tf.float32)
    pos_a = tf.constant(0.0, shape=(1,4), dtype=tf.float32)
    neg_pc = tf.constant(0.1, shape=(1,1), dtype=tf.float32)
    return pos_pc, pos_label, pos_pr, pos_gt_bbox, pos_a, neg_pc


def bbox_encode(gt_bbox, anchors, normlization):
    # Dummy encoding returns zeros same shape as gt_bbox
    return tf.zeros_like(gt_bbox)


def smooth_l1_loss(x):
    # Dummy smooth l1 loss as absolute value
    return tf.abs(x)


def focal_loss(pos_pc, pos_label, neg_pc, alpha=0.25, gamma=2.0):
    # Dummy focal loss returns scalar loss value
    return tf.constant(1.0)


class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.num_classes = config['num_classes']
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if self.mode == 'train' else 1
        self.lr = config['lr']
        self.backone_name = config.get('backone', 'resnetv1_18')

        # Backbone placeholder
        self.backone = lambda img: dummy_backbone_model(
            img,
            config.get('backone_conv_trainable', True),
            config.get('backone_bn_trainable', True),
            None)

        # For weight saving/loading compatibility, build the keras.Model once, stored in self.model
        image_input = tf.keras.Input(shape=[None, None, 3], batch_size=self.batch_size, dtype=tf.float32)

        if self.mode == 'train':
            gt_input = tf.keras.Input(shape=[None, 5], batch_size=self.batch_size, dtype=tf.float32)
            outputs = self._build_graph(image_input, gt_input)
            self.model = tf.keras.Model(inputs=[image_input, gt_input], outputs=outputs, name='retinanet')
        else:
            outputs = self._build_graph(image_input)
            self.model = tf.keras.Model(inputs=image_input, outputs=outputs, name='retinanet')

    def _build_graph(self, image, gt=None):
        num_fpn_layers = 5
        fpn_channels = 256

        # Get backbone features
        endpoints = self.backone(image)
        # endpoints dummy returns more than needed but fpn_generator expects last 5 features
        fpn = fpn_generator(endpoints[1:], fpn_channels, num_fpn_layers, mode='dconv')
        p3, p4, p5, p6, p7 = fpn

        # Anchors setup similar to original code
        dw_rate = [8., 16., 32., 64., 128.]
        anchors = [
            [[4, 4], [8, 2], [2, 8], [12, 4], [4/3., 12.]],
            [[4, 4], [8, 2], [2, 8], [12, 4], [4/3., 12.]],
            [[4, 4], [8, 2], [2, 8], [12, 4], [4/3., 12.]],
            [[4, 4], [8, 2], [2, 8], [12, 4], [4/3., 12.]],
            [[4, 4], [8, 2], [2, 8], [12, 4], [4/3., 12.]],
        ]

        # Generate anchors for each feature map
        anchors_all_list = anchor_generator(fpn, anchors, dw_rate, flatten=True)
        anchors_all = tf.concat(anchors_all_list, axis=0)

        # Use fused _head for classification and regression heads
        # Create heads models if not exist yet
        if not hasattr(self, 'head_model'):
            self.head_model = self._head(fpn_channels, 5)

        # Pass each FPN level through head model
        cla_preds = []
        reg_preds = []
        for feature in fpn:
            cla_pred, reg_pred = self.head_model(feature)
            cla_preds.append(cla_pred)
            reg_preds.append(reg_pred)

        pc = tf.concat(cla_preds, axis=1)  # (batch, total_anchors, num_classes)
        pr = tf.concat(reg_preds, axis=1)  # (batch, total_anchors, 4)

        pc = tf.nn.sigmoid(pc)

        if self.mode == 'train':
            # Compute loss by looping over batch dimension
            loss = tf.constant([0., 0.], dtype=tf.float32, shape=[1, 2])
            i = tf.constant(0)
            cond = lambda loss, i: tf.less(i, self.batch_size)

            def body(loss, i):
                gt_i = tf.gather(gt, i)
                pc_i = tf.gather(pc, i)
                pr_i = tf.gather(pr, i)
                loss_i = self._compute_one_image_loss(gt_i, anchors_all, pc_i, pr_i)
                return tf.add(loss, loss_i), tf.add(i, 1)

            loss, _ = tf.while_loop(cond, body, (loss, i))
            loss /= tf.cast(self.batch_size, tf.float32)
            return loss  # Loss tensor as output

        else:
            # Inference mode: Generate detections applying NMS
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
                scores_i = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                bbox_i = tf.boolean_mask(y1x1y2x2, filter_mask[:, i])
                selected_indices = tf.image.non_max_suppression(
                    bbox_i, scores_i, nms_max_boxes, nms_iou_threshold)
                scores.append(tf.gather(scores_i, selected_indices))
                bbox.append(tf.gather(bbox_i, selected_indices))
                class_id.append(
                    tf.ones_like(tf.gather(scores_i, selected_indices), tf.int32) * (i + 1))
            bbox = tf.concat(bbox, axis=0)
            scores = tf.concat(scores, axis=0)
            class_id = tf.concat(class_id, axis=0)
            detection_pred = [scores, bbox, class_id]
            return detection_pred

    def _compute_one_image_loss(self, gt, anchors, pc, pr):
        # Extract valid ground-truth entries based on argmin logic (dummy safe guard)
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

        reg_loss = tf.reduce_sum(smooth_l1_loss(pos_pr - pos_gr))

        pos_label_one_hot = tf.one_hot(pos_label, self.num_classes, dtype=tf.float32)

        cla_loss = focal_loss(pos_pc, pos_label_one_hot, neg_pc, alpha=0.25, gamma=2.)

        reg_loss = tf.reshape(reg_loss, [1, 1])
        cla_loss = tf.reshape(cla_loss, [1, 1])

        loss = tf.concat([cla_loss, reg_loss], axis=-1)
        return loss

    def _head(self, input_channels, anchors):
        x = tf.keras.Input(shape=[None, None, input_channels], dtype=tf.float32)

        # Classification branch
        conv1 = layers.Conv2D(256, 3, 1, 'same',
                              kernel_initializer='he_normal', name='cla_conv1')(x)
        bn1 = layers.BatchNormalization(epsilon=1.001e-5, name='cla_bn1')(conv1)
        relu1 = layers.Activation('relu', name='cla_relu1')(bn1)

        conv1 = layers.Conv2D(256, 3, 1, 'same',
                              kernel_initializer='he_normal', name='cla_conv2')(relu1)
        bn1 = layers.BatchNormalization(epsilon=1.001e-5, name='cla_bn2')(conv1)
        relu1 = layers.Activation('relu', name='cla_relu2')(bn1)

        conv1 = layers.Conv2D(256, 3, 1, 'same',
                              kernel_initializer='he_normal', name='cla_conv3')(relu1)
        bn1 = layers.BatchNormalization(epsilon=1.001e-5, name='cla_bn3')(conv1)
        relu1 = layers.Activation('relu', name='cla_relu3')(bn1)

        cla_pred = layers.Conv2D(self.num_classes * anchors, 3, 1, 'same',
                                 kernel_initializer='he_normal',
                                 name='cla_conv4',
                                 bias_initializer=tf.constant_initializer(-4.595))(relu1)
        cla_pred = tf.reshape(cla_pred, [self.batch_size, -1, self.num_classes])

        # Regression branch
        conv2 = layers.Conv2D(256, 3, 1, 'same',
                              kernel_initializer='he_normal', name='reg_conv1')(x)
        bn2 = layers.BatchNormalization(epsilon=1.001e-5, name='reg_bn1')(conv2)
        relu2 = layers.Activation('relu', name='reg_relu1')(bn2)

        conv2 = layers.Conv2D(256, 3, 1, 'same',
                              kernel_initializer='he_normal', name='reg_conv2')(relu2)
        bn2 = layers.BatchNormalization(epsilon=1.001e-5, name='reg_bn2')(conv2)
        relu2 = layers.Activation('relu', name='reg_relu2')(bn2)

        conv2 = layers.Conv2D(256, 3, 1, 'same',
                              kernel_initializer='he_normal', name='reg_conv3')(relu2)
        bn2 = layers.BatchNormalization(epsilon=1.001e-5, name='reg_bn3')(conv2)
        relu2 = layers.Activation('relu', name='reg_relu3')(bn2)

        reg_pred = layers.Conv2D(4 * anchors, 3, 1, 'same',
                                 kernel_initializer='he_normal', name='reg_conv4')(relu2)
        reg_pred = tf.reshape(reg_pred, [self.batch_size, -1, 4])

        return tf.keras.Model(inputs=x, outputs=[cla_pred, reg_pred])

    def save_weights(self, filepath, overwrite=True, save_format=None):
        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name)


def my_model_function():
    # Example config for training usage with batch_size=2, learning rate=1e-3
    config = {
        'num_classes': 20,
        'batch_size': 2,
        'mode': 'train',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'backone': 'resnetv1_18',
        'backone_conv_trainable': True,
        'backone_bn_trainable': True,
    }
    return MyModel(config)


def GetInput():
    # Generate dummy input image of shape (batch_size, 256, 256, 3)
    # Matches expected input tensor shape for MyModel
    batch_size = 2
    height = 256
    width = 256
    dtype = tf.float32
    return tf.random.uniform((batch_size, height, width, 3), dtype=dtype)

