from tensorflow import keras

import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_6.2090354') # path to the SavedModel directory
tflite_model = converter.convert()

def _traditional_nms(self, boxes, mask_coef, scores, iou_threshold=0.5, score_threshold=0.3, max_class_output_size=100, max_output_size=300, soft_nms_sigma=0.5):
        num_classes = tf.shape(scores)[1]

        _num_coef = tf.shape(mask_coef)[1]
        _boxes = tf.zeros((max_class_output_size*num_classes, 4), tf.float32)
        _coefs = tf.zeros((max_class_output_size*num_classes, _num_coef), tf.float32)
        _classes = tf.zeros((max_class_output_size*num_classes), tf.float32)
        _scores = tf.zeros((max_class_output_size*num_classes), tf.float32)

        for _cls in range(num_classes):
            cls_scores = scores[:, _cls]
            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                boxes, 
                cls_scores, 
                max_output_size=max_class_output_size, 
                iou_threshold=iou_threshold, 
                score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma)

            _update_boxes = tf.gather(boxes, selected_indices)
            _num_boxes = tf.shape(_update_boxes)[0]
            _ind_boxes = tf.range(_cls*max_class_output_size, _cls*max_class_output_size+_num_boxes)

            _boxes = tf.tensor_scatter_nd_update(_boxes, tf.expand_dims(_ind_boxes, axis=-1), _update_boxes)
            _coefs = tf.tensor_scatter_nd_update(_coefs, tf.expand_dims(_ind_boxes, axis=-1), tf.gather(mask_coef, selected_indices))
            _classes = tf.tensor_scatter_nd_update(_classes, tf.expand_dims(_ind_boxes, axis=-1), tf.gather(cls_scores, selected_indices) * 0.0 + tf.cast(_cls, dtype=tf.float32) + 1.0)
            _scores = tf.tensor_scatter_nd_update(_scores, tf.expand_dims(_ind_boxes, axis=-1), tf.gather(cls_scores, selected_indices))

        _ids = tf.argsort(_scores, direction='DESCENDING')
        scores = tf.gather(_scores, _ids)[:max_output_size]
        boxes = tf.gather(_boxes, _ids)[:max_output_size]
        mask_coef = tf.gather(_coefs, _ids)[:max_output_size]
        classes = tf.gather(_classes, _ids)[:max_output_size]

        return boxes, mask_coef, classes, scores

def __call__(self, net_outs, trad_nms=True):
        """
        Args:
             pred_offset: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            pred_cls: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            pred_mask_coef: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            priors: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_out: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.
            Note that the outputs are sorted only if cross_class_nms is False
        """

        box_p = net_outs['pred_offset']  # [1, 27429, 4]
        class_p = net_outs['pred_cls']  # [1, 27429, 2]
        coef_p = net_outs['pred_mask_coef']  # [1, 27429, 32]
        anchors = net_outs['priors']  # [27429, 4]
        proto_p = net_outs['proto_out']  # [1, 90, 302, 32]
        
        proto_h = tf.shape(proto_p)[1]
        proto_w = tf.shape(proto_p)[2]

        box_decode = self._decode(box_p, anchors)  # [1, 27429, 4]
        
        num_class = tf.shape(class_p)[2] - 1

        # Apply softmax to the prediction class
        class_p = tf.nn.softmax(class_p, axis=-1)
        # exclude the background class
        class_p = class_p[:, :, 1:]
        # get the max score class of 27429 predicted boxes
        class_p_max = tf.reduce_max(class_p, axis=-1)  # [1, 27429]
        batch_size = tf.shape(class_p_max)[0]

        detection_boxes = tf.zeros((batch_size, self.max_output_size, 4), tf.float32)
        detection_classes = tf.zeros((batch_size, self.max_output_size), tf.float32)
        detection_scores = tf.zeros((batch_size, self.max_output_size), tf.float32)
        detection_masks = tf.zeros((batch_size, self.max_output_size, proto_h, proto_w), tf.float32)
        num_detections = tf.zeros((batch_size), tf.int32)

        for b in range(batch_size):
            # filter predicted boxes according the class score
            class_thre = tf.boolean_mask(class_p[b], class_p_max[b] > 0.3)
            box_thre = tf.boolean_mask(box_decode[b], class_p_max[b] > 0.3) 
            coef_thre = tf.boolean_mask(coef_p[b], class_p_max[b] > 0.3)

            if tf.size(class_thre) != 0:
                if not trad_nms:
                    box_thre, coef_thre, class_ids, class_thre = _fast_nms(box_thre, coef_thre, class_thre)
                else:
                    box_thre, coef_thre, class_ids, class_thre = self._traditional_nms(box_thre, coef_thre, class_thre)

                # Padding with zeroes to reach max_output_size
                class_ids = tf.concat([class_ids, tf.zeros(self.max_output_size - tf.shape(box_thre)[0])], 0)
                class_thre = tf.concat([class_thre, tf.zeros(self.max_output_size - tf.shape(box_thre)[0])], 0)
                num_detection = [tf.shape(box_thre)[0]]
                pad_num_detection = self.max_output_size - num_detection[0]

                _masks_coef = tf.matmul(proto_p[b], tf.transpose(coef_thre))
                _masks_coef = tf.sigmoid(_masks_coef) # [138, 138, NUM_BOX]

                boxes, masks = self._sanitize(_masks_coef, box_thre)
                masks = tf.transpose(masks, (2,0,1))
                paddings = tf.convert_to_tensor( [[0, pad_num_detection], [0,0], [0, 0]])
                masks = tf.pad(masks, paddings, "CONSTANT")
                
                paddings = tf.convert_to_tensor( [[0, pad_num_detection], [0, 0]])
                boxes = tf.pad(boxes, paddings, "CONSTANT")

                detection_boxes = tf.tensor_scatter_nd_update(detection_boxes, [[b]], tf.expand_dims(boxes, 0))
                detection_classes = tf.tensor_scatter_nd_update(detection_classes, [[b]], tf.expand_dims(class_ids, 0))
                detection_scores = tf.tensor_scatter_nd_update(detection_scores, [[b]], tf.expand_dims(class_thre, 0))
                detection_masks = tf.tensor_scatter_nd_update(detection_masks, [[b]], tf.expand_dims(masks, 0))
                num_detections = tf.tensor_scatter_nd_update(num_detections, [[b]], num_detection)
        
        result = {'detection_boxes': detection_boxes,'detection_classes': detection_classes, 'detection_scores': detection_scores, 'detection_masks': detection_masks, 'num_detections': num_detections}
        return result

import tensorflow as tf
import os
import cv2

IMAGE_PATH = '/home/deploy/ved/test/'

train_images = tf.keras.preprocessing.image_dataset_from_directory(
        directory=IMAGE_PATH, labels='inferred', label_mode='int', class_names=None,
        color_mode='rgb', batch_size=1)

def representative_dataset():
  for i in range(100):
    image = train_images[i]
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [720, 2410])
    image = tf.cast(image / 255., tf.float64)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/saved_model_0.6849702')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
			tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
			tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
			tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
			]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_quant_model = converter.convert()

open("yolact.tflite", "wb").write(tflite_quant_model)