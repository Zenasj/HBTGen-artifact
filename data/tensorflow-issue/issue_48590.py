# tf.random.uniform((batch_size, num_priors, 4), dtype=tf.float32)
# Note: Input is assumed to be a dictionary-like structure with tensors for keys:
#   'pred_offset': [batch, num_priors, 4]
#   'pred_cls': [batch, num_priors, num_classes+1]
#   'pred_mask_coef': [batch, num_priors, mask_dim]
#   'priors': [num_priors, 4]
#   'proto_out': [batch, mask_h, mask_w, mask_dim]
#
# The model includes a custom traditional NMS implementation using tf.tensor_scatter_nd_update,
# which is known to cause TFLite conversion issues due to unsupported ops.
# This code reconstructs the core logic and usage with inferred shapes and types,
# including a fused MyModel class that provides a __call__ method wrapping the detection logic.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, max_output_size=300, max_class_output_size=100, iou_threshold=0.5, score_threshold=0.3, soft_nms_sigma=0.5):
        super().__init__()
        self.max_output_size = max_output_size
        self.max_class_output_size = max_class_output_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms_sigma = soft_nms_sigma

    def _traditional_nms(self, boxes, mask_coef, scores,
                         iou_threshold=None, score_threshold=None,
                         max_class_output_size=None, max_output_size=None,
                         soft_nms_sigma=None):
        # Defaults if parameters are None
        iou_threshold = iou_threshold if iou_threshold is not None else self.iou_threshold
        score_threshold = score_threshold if score_threshold is not None else self.score_threshold
        max_class_output_size = max_class_output_size if max_class_output_size is not None else self.max_class_output_size
        max_output_size = max_output_size if max_output_size is not None else self.max_output_size
        soft_nms_sigma = soft_nms_sigma if soft_nms_sigma is not None else self.soft_nms_sigma

        num_classes = tf.shape(scores)[1]
        num_coef = tf.shape(mask_coef)[1]

        _boxes = tf.zeros((max_class_output_size * num_classes, 4), tf.float32)
        _coefs = tf.zeros((max_class_output_size * num_classes, num_coef), tf.float32)
        _classes = tf.zeros((max_class_output_size * num_classes), tf.float32)
        _scores = tf.zeros((max_class_output_size * num_classes), tf.float32)

        for _cls in tf.range(num_classes):
            cls_scores = scores[:, _cls]  # [num_priors]
            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                boxes,
                cls_scores,
                max_output_size=max_class_output_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma)

            _update_boxes = tf.gather(boxes, selected_indices)
            num_selected = tf.shape(_update_boxes)[0]
            _ind_boxes = tf.range(_cls * max_class_output_size, _cls * max_class_output_size + num_selected)

            # Update tensors via tensor_scatter_nd_update
            _boxes = tf.tensor_scatter_nd_update(_boxes,
                                                tf.expand_dims(_ind_boxes, axis=-1),
                                                _update_boxes)
            _coefs = tf.tensor_scatter_nd_update(_coefs,
                                                tf.expand_dims(_ind_boxes, axis=-1),
                                                tf.gather(mask_coef, selected_indices))
            _classes = tf.tensor_scatter_nd_update(_classes,
                                                  tf.expand_dims(_ind_boxes, axis=-1),
                                                  tf.gather(cls_scores, selected_indices) * 0.0 + tf.cast(_cls, tf.float32) + 1.0)
            _scores = tf.tensor_scatter_nd_update(_scores,
                                                 tf.expand_dims(_ind_boxes, axis=-1),
                                                 tf.gather(cls_scores, selected_indices))

        _ids = tf.argsort(_scores, direction='DESCENDING')
        boxes_out = tf.gather(_boxes, _ids)[:max_output_size]
        mask_coef_out = tf.gather(_coefs, _ids)[:max_output_size]
        classes_out = tf.gather(_classes, _ids)[:max_output_size]
        scores_out = tf.gather(_scores, _ids)[:max_output_size]

        return boxes_out, mask_coef_out, classes_out, scores_out

    def __call__(self, net_outs, trad_nms=True):
        """
        Args:
            net_outs: dict with keys:
                'pred_offset': [batch, num_priors, 4] tensor of box offsets
                'pred_cls': [batch, num_priors, num_classes+1] tensor (includes background class)
                'pred_mask_coef': [batch, num_priors, mask_dim]
                'priors': [num_priors, 4]
                'proto_out': [batch, mask_h, mask_w, mask_dim]

            trad_nms: bool, whether to use traditional NMS or a faster variant (not implemented here)

        Returns:
            dict of detection outputs with the following keys and shapes:
                'detection_boxes': [batch, max_output_size, 4]
                'detection_classes': [batch, max_output_size]
                'detection_scores': [batch, max_output_size]
                'detection_masks': [batch, max_output_size, mask_h, mask_w]
                'num_detections': [batch]
        """
        # Unpack inputs
        box_p = net_outs['pred_offset']  # [batch, num_priors, 4]
        class_p = net_outs['pred_cls']  # [batch, num_priors, num_classes+1]
        coef_p = net_outs['pred_mask_coef']  # [batch, num_priors, mask_dim]
        anchors = net_outs['priors']  # [num_priors, 4]
        proto_p = net_outs['proto_out']  # [batch, mask_h, mask_w, mask_dim]

        proto_h = tf.shape(proto_p)[1]
        proto_w = tf.shape(proto_p)[2]

        # Decode boxes: here we just assume a simple addition for demo (real decode logic should be implemented)
        # In practice, decoding often happens as:
        # decoded_boxes = decode_boxes(box_p, anchors)
        box_decode = box_p + tf.expand_dims(anchors, 0)  # [batch, num_priors, 4]

        num_class = tf.shape(class_p)[2] - 1  # Exclude background class
        class_p = tf.nn.softmax(class_p, axis=-1)  # Softmax over classes
        class_p = class_p[:, :, 1:]  # Exclude background class probabilities

        class_p_max = tf.reduce_max(class_p, axis=-1)  # Max class score per box [batch, num_priors]
        batch_size = tf.shape(class_p_max)[0]

        detection_boxes = tf.zeros((batch_size, self.max_output_size, 4), tf.float32)
        detection_classes = tf.zeros((batch_size, self.max_output_size), tf.float32)
        detection_scores = tf.zeros((batch_size, self.max_output_size), tf.float32)
        detection_masks = tf.zeros((batch_size, self.max_output_size, proto_h, proto_w), tf.float32)
        num_detections = tf.zeros((batch_size,), tf.int32)

        # For-loop over batch dimension (small batch sizes typical)
        for b in tf.range(batch_size):
            # Filter boxes above a minimal confidence threshold
            class_thre = tf.boolean_mask(class_p[b], class_p_max[b] > self.score_threshold)
            box_thre = tf.boolean_mask(box_decode[b], class_p_max[b] > self.score_threshold)
            coef_thre = tf.boolean_mask(coef_p[b], class_p_max[b] > self.score_threshold)

            def empty_result():
                # Returns empty tensors with shapes matching expected output padded with zeros
                empty_boxes = tf.zeros((0, 4), tf.float32)
                empty_coefs = tf.zeros((0, tf.shape(coef_p)[2]), tf.float32)
                empty_classes = tf.zeros((0,), tf.float32)
                empty_scores = tf.zeros((0,), tf.float32)
                return empty_boxes, empty_coefs, empty_classes, empty_scores

            boxes_nms, coefs_nms, class_ids, class_scores = tf.cond(
                tf.size(class_thre) > 0,
                lambda: self._traditional_nms(box_thre, coef_thre, class_thre),
                empty_result)

            # Pad results to max_output_size so that tensor shapes are consistent
            pad_size = self.max_output_size - tf.shape(boxes_nms)[0]
            pad_boxes = tf.pad(boxes_nms, [[0, pad_size], [0, 0]], constant_values=0)
            pad_coefs = tf.pad(coefs_nms, [[0, pad_size], [0, 0]], constant_values=0)
            pad_classes = tf.pad(class_ids, [[0, pad_size]], constant_values=0)
            pad_scores = tf.pad(class_scores, [[0, pad_size]], constant_values=0)

            num_det = tf.shape(boxes_nms)[0]

            # Calculate masks = sigmoid(proto_p @ coefs^T)
            masks_coef = tf.matmul(proto_p[b], tf.transpose(pad_coefs))  # [mask_h, mask_w, max_output_size]
            masks_coef = tf.sigmoid(masks_coef)
            # Transpose to [max_output_size, mask_h, mask_w] for output format
            masks = tf.transpose(masks_coef, perm=[2, 0, 1])

            # Pad masks to ensure fixed size (already max_output_size in dim 0)
            # No extra padding needed since masks_coef matches max_output_size padded above

            # Update batch slice in output tensors via scatter update
            detection_boxes = tf.tensor_scatter_nd_update(
                detection_boxes, [[b]], tf.expand_dims(pad_boxes, 0))
            detection_classes = tf.tensor_scatter_nd_update(
                detection_classes, [[b]], tf.expand_dims(pad_classes, 0))
            detection_scores = tf.tensor_scatter_nd_update(
                detection_scores, [[b]], tf.expand_dims(pad_scores, 0))
            detection_masks = tf.tensor_scatter_nd_update(
                detection_masks, [[b]], tf.expand_dims(masks, 0))
            num_detections = tf.tensor_scatter_nd_update(
                num_detections, [[b]], [num_det])

        result = {
            'detection_boxes': detection_boxes,
            'detection_classes': detection_classes,
            'detection_scores': detection_scores,
            'detection_masks': detection_masks,
            'num_detections': num_detections
        }
        return result

def my_model_function():
    # Return an instance of the MyModel with default parameters
    return MyModel()

def GetInput():
    # Returns a dictionary input matching expected model input with dummy random data
    batch_size = 1
    num_priors = 27429  # Example number of priors from issue comments
    num_classes_plus_one = 3  # Example: background + 2 classes
    mask_dim = 32
    mask_h = 90
    mask_w = 302

    inputs = {
        'pred_offset': tf.random.uniform((batch_size, num_priors, 4), dtype=tf.float32),
        'pred_cls': tf.random.uniform((batch_size, num_priors, num_classes_plus_one), dtype=tf.float32),
        'pred_mask_coef': tf.random.uniform((batch_size, num_priors, mask_dim), dtype=tf.float32),
        'priors': tf.random.uniform((num_priors, 4), dtype=tf.float32),
        'proto_out': tf.random.uniform((batch_size, mask_h, mask_w, mask_dim), dtype=tf.float32)
    }
    return inputs

