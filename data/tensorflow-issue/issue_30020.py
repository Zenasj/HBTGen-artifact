# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape (batch, height, width, channels)

import tensorflow as tf

def smooth_l1_loss(y_true, y_pred, config=None):
    # Implement a smooth L1 loss similar to Fast R-CNN
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * tf.square(diff)) + (1 - less_than_one) * (diff - 0.5)
    return loss

def focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2.0):
    # Basic focal loss implementation for classification
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)

    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent

# Placeholder norm boxes and rboxes normalization (identity for TF2 compatibility)
def norm_boxes_graph(boxes, shape):
    # Normalize bounding boxes to [0,1], input shape assumed at least 3 dims [H, W, C]
    h, w = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
    # box format assumed [N, (y1,x1,y2,x2)]
    boxes = tf.cast(boxes, tf.float32)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
    y1 /= h
    y2 /= h
    x1 /= w
    x2 /= w
    return tf.concat([y1, x1, y2, x2], axis=-1)

def norm_rboxes_graph(rboxes, shape):
    # Similar normalization for rotated boxes, dummy identity here 
    return rboxes

def build_anchors(config, image_shape=None, norm=True, to_tensor=True):
    # Return dummy anchors tensor - assume [num_anchors, 4]
    anchors = tf.random.uniform((1000, 4), dtype=tf.float32)
    if norm:
        anchors = norm_boxes_graph(anchors, image_shape[:2])
    if to_tensor:
        anchors = tf.convert_to_tensor(anchors)
    return anchors

def parse_image_meta_graph(meta):
    # Returns dict with 'active_class_ids'
    # For simplicity, assume input is a tensor and output [batch, num_classes] ones
    batch_size = tf.shape(meta)[0]
    num_classes = 80 # example COCO classes
    active_class_ids = tf.ones((batch_size, num_classes), dtype=tf.int32)
    return {"active_class_ids": active_class_ids}

class Loss_tower():
    def __init__(self, config):
        self.config = config

    def rpn_cla_loss(self, rpn_match, rpn_class_logits):
        rpn_match = tf.squeeze(rpn_match, -1)
        anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
        indices = tf.where(tf.not_equal(rpn_match, 0))
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        if self.config.RPN_CLASS_LOSS_TYPE == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_class, logits=rpn_class_logits)
        else:  # focal_loss
            loss = focal_loss(prediction_tensor=rpn_class_logits,
                              target_tensor=tf.cast(tf.one_hot(anchor_class, 2), tf.float32))
        loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
        return loss

    @staticmethod
    def batch_pack_graph(x, counts, num_rows):
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)

    def rpn_bbox_loss(self, target_bbox, rpn_match, rpn_bbox):
        rpn_match = tf.squeeze(rpn_match, -1)
        indices = tf.where(tf.equal(rpn_match, 1))
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)
        batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = self.batch_pack_graph(target_bbox, batch_counts, self.config.IMAGES_PER_GPU)
        loss = smooth_l1_loss(y_true=target_bbox, y_pred=rpn_bbox, config=self.config)
        loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
        return loss

    def mrcnn_cla_loss(self, target_class_ids, pred_class_logits, active_class_ids):
        target_class_ids = tf.cast(target_class_ids, tf.int64)
        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        pred_active = tf.gather(active_class_ids[0], pred_class_ids)
        if self.config.RCNN_CLASS_LOSS_TYPE == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
        else:  # focal_loss
            loss = focal_loss(prediction_tensor=pred_class_logits,
                              target_tensor=tf.cast(tf.one_hot(target_class_ids, tf.cast(active_class_ids[1][0], tf.int32)), tf.float32))
            loss = tf.reduce_sum(loss, axis=-1)
        loss *= tf.cast(pred_active, loss.dtype)
        loss = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(pred_active, loss.dtype))
        return loss

    def mrcnn_bbox_loss(self, target_bbox, target_class_ids, pred_bbox):
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        target_bbox = tf.reshape(target_bbox, (-1, 4))
        pred_bbox = tf.reshape(pred_bbox, (-1, pred_bbox.shape.as_list()[2], 4))
        positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)
        loss = smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox, config=self.config) if tf.size(target_bbox) > 0 else tf.constant(0.0)
        loss = tf.reduce_mean(loss)
        return loss

    def mrcnn_mask_loss(self, target_masks, target_class_ids, pred_masks):
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        if self.config.MASK_LOSS_TYPE == "binary_ce":
            pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
            positive_ix = tf.where(target_class_ids > 0)[:, 0]
            positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
            indices = tf.stack([positive_ix, positive_class_ids], axis=1)
            y_true = tf.gather(target_masks, positive_ix)
            y_pred = tf.gather_nd(pred_masks, indices)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred) if tf.size(y_true) > 0 else tf.constant(0.0)
        else:  # focal_loss
            positive_ix = tf.where(target_class_ids > 0)[:, 0]
            y_true = tf.gather(target_masks, positive_ix)
            y_pred = tf.gather(pred_masks, positive_ix)
            if tf.size(y_true) > 0:
                loss = focal_loss(y_pred, tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), self.config.NUM_CLASSES), tf.float32))
            else:
                loss = tf.constant(0.0)
        loss = tf.reduce_mean(loss)
        return loss

# Dummy Config class:
class Config:
    # assume defaults matching the snippets
    def __init__(self):
        # input image shape H,W,C
        self.IMAGE_SHAPE = (128, 128, 3)
        self.BACKBONE = "resnet101"
        self.USE_RPN_ROIS = True
        self.BATCH_SIZE = 2
        self.IMAGES_PER_GPU = 2
        self.NUM_CLASSES = 81  # 80 + background
        self.MODEL_DIR = './model_ckpt'
        self.RPN_CLASS_LOSS_TYPE = 'cross_entropy'
        self.RCNN_CLASS_LOSS_TYPE = 'cross_entropy'
        self.MASK_LOSS_TYPE = 'binary_ce'

# Placeholder backbone and heads for demonstration. These are simple stubs.
class Backbone(tf.keras.layers.Layer):
    def call(self, x):
        # Return feature maps C2,C3,C4,C5 as simple scaled inputs
        C2 = tf.image.resize(x, (32, 32))
        C3 = tf.image.resize(x, (16, 16))
        C4 = tf.image.resize(x, (8, 8))
        C5 = tf.image.resize(x, (4, 4))
        return C2, C3, C4, C5

class FPN(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs: list of feature maps from backbone
        # Return pyramid levels (P2 through P6)
        C2, C3, C4, C5 = inputs
        P2 = C2  # just placeholders
        P3 = C3
        P4 = C4
        P5 = C5
        P6 = tf.image.resize(C5, (2, 2))
        return P2, P3, P4, P5, P6

class RPN(tf.keras.layers.Layer):
    def call(self, feature_maps):
        # Return dummy outputs:
        batch_size = tf.shape(feature_maps[0])[0]
        num_anchors = 1000
        rpn_class_logits = tf.random.uniform([batch_size, num_anchors, 2])
        rpn_class = tf.random.uniform([batch_size, num_anchors], maxval=2, dtype=tf.int32)
        rpn_bbox = tf.random.uniform([batch_size, num_anchors, 4])
        return rpn_class_logits, rpn_class, rpn_bbox

class ProposalLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs: rpn_class, rpn_bbox, anchors
        # Return dummy ROIS: [batch, num_rois, 4]
        batch_size = tf.shape(inputs[0])[0]
        num_rois = 200
        return tf.random.uniform([batch_size, num_rois, 4])

class Classifier(tf.keras.layers.Layer):
    def call(self, inputs, training=False):
        # inputs: rois, image_meta, feature_maps
        batch_size = tf.shape(inputs[0])[0]
        num_rois = tf.shape(inputs[0])[1]
        num_classes = 81
        mrcnn_class_logits = tf.random.uniform([batch_size, num_rois, num_classes])
        mrcnn_bbox = tf.random.uniform([batch_size, num_rois, num_classes, 4])
        return {"mrcnn_class_logits": mrcnn_class_logits, "mrcnn_bbox": mrcnn_bbox}

class Masker(tf.keras.layers.Layer):
    def call(self, inputs, training=False):
        # inputs: rois, image_meta, feature_maps
        batch_size = tf.shape(inputs[0])[0]
        num_rois = tf.shape(inputs[0])[1]
        height, width = 28, 28
        num_classes = 81
        return tf.random.uniform([batch_size, num_rois, height, width, num_classes])

class Detect(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs: rois, class_logits, bbox, image_meta
        batch_size = tf.shape(inputs[0])[0]
        num_detections = 100
        # Each detection: y1,x1,y2,x2,class_id,score
        detections = tf.random.uniform([batch_size, num_detections, 6])
        return detections

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.mode = 'training'  # or 'inference'
        self.backbone = Backbone()
        self.fpn = FPN()
        self.rpn = RPN()
        self.proposal = ProposalLayer()
        self.classifier = Classifier()
        self.masker = Masker()
        self.detect = Detect()

    def call(self, inputs, training=True):
        # inputs is a tuple or list
        # Unpack inputs depending on mode
        if self.mode == "training":
            (input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, 
             input_gt_rboxes, input_gt_global_mask, input_gt_masks, input_gt_masks_score, 
             input_rpn_match, input_rpn_bbox) = inputs
        else:
            input_image, input_image_meta = inputs

        gt_boxes = norm_boxes_graph(input_gt_boxes, tf.shape(input_image)[1:3])
        gt_rboxes = norm_rboxes_graph(input_gt_rboxes, tf.shape(input_image)[1:3])

        if self.config.BACKBONE == "resnet101":
            C2, C3, C4, C5 = self.backbone(input_image)
        else:
            C2 = C3 = C4 = C5 = input_image  # fallback

        P2, P3, P4, P5, P6 = self.fpn([C2, C3, C4, C5])
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        rpn_class_logits, rpn_class, rpn_bbox = self.rpn(rpn_feature_maps)
        anchors = build_anchors(self.config, image_shape=self.config.IMAGE_SHAPE, norm=True, to_tensor=True)
        rpn_rois = self.proposal([rpn_class, rpn_bbox, anchors])

        if self.mode == "training":
            active_class_ids = parse_image_meta_graph(input_image_meta)["active_class_ids"]
            target_rois = rpn_rois if self.config.USE_RPN_ROIS else norm_boxes_graph(input_rpn_bbox, input_image.shape.as_list()[1:3])

            # Detect target is a placeholder for target data preparation
            # Here we just simulate outputs
            output_rois = target_rois
            target_class_ids = input_gt_class_ids
            target_bbox = input_rpn_bbox
            target_mask = input_gt_masks  # assumed variable
            target_embed_length = None  # placeholder
            target_rbox = None

            mrcnn_box_outputs = self.classifier([output_rois, input_image_meta, mrcnn_feature_maps], True)
            mrcnn_mask = self.masker([output_rois, input_image_meta, mrcnn_feature_maps], True)

            return (rpn_class_logits, rpn_bbox, mrcnn_box_outputs, mrcnn_mask, 
                    target_class_ids, target_bbox, target_mask, active_class_ids)
        else:
            mrcnn_box_outputs = self.classifier([rpn_rois, input_image_meta, mrcnn_feature_maps])
            detections = self.detect([rpn_rois, mrcnn_box_outputs["mrcnn_class_logits"], 
                                     mrcnn_box_outputs["mrcnn_bbox"], input_image_meta])
            mrcnn_mask = self.masker([detections[..., :4], input_image_meta, mrcnn_feature_maps])
            return detections, mrcnn_mask

def my_model_function():
    return MyModel()

def GetInput():
    config = Config()
    batch = config.BATCH_SIZE
    H, W, C = config.IMAGE_SHAPE
    # Compose inputs for training mode
    
    # input_image float32 tensor [batch, H, W, C]
    input_image = tf.random.uniform((batch, H, W, C), dtype=tf.float32)
    # input_image_meta float32 tensor shape [batch, 12] (example)
    input_image_meta = tf.random.uniform((batch, 12), dtype=tf.float32)
    # input_gt_class_ids int32 tensor [batch, num_gts], num_gts arbitrary, padded to 10 here
    input_gt_class_ids = tf.random.uniform((batch, 10), maxval=80, dtype=tf.int32)
    # input_gt_boxes float32 [batch, num_gts, 4]
    input_gt_boxes = tf.random.uniform((batch, 10, 4), dtype=tf.float32)
    # input_gt_rboxes float32 [batch, num_gts, 5]
    input_gt_rboxes = tf.random.uniform((batch, 10, 5), dtype=tf.float32)
    # input_gt_global_mask boolean [batch, 10]
    input_gt_global_mask = tf.cast(tf.random.uniform((batch, 10), maxval=2, dtype=tf.int32), tf.bool)
    # input_gt_masks boolean [batch, 10, 28, 28]
    input_gt_masks = tf.cast(tf.random.uniform((batch, 10, 28, 28), maxval=2, dtype=tf.int32), tf.bool)
    # input_gt_masks_score float32 [batch, 10]
    input_gt_masks_score = tf.random.uniform((batch, 10), dtype=tf.float32)
    # input_rpn_match int32 [batch, num_anchors, 1], here num_anchors = 1000 with 1 dim
    input_rpn_match = tf.random.uniform((batch, 1000, 1), maxval=3, dtype=tf.int32) - 1
    # input_rpn_bbox float32 [batch, 1000, 4]
    input_rpn_bbox = tf.random.uniform((batch, 1000, 4), dtype=tf.float32)

    return (input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_rboxes,
            input_gt_global_mask, input_gt_masks, input_gt_masks_score, input_rpn_match,
            input_rpn_bbox)

