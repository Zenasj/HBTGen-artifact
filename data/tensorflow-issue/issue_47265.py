import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from typing import Tuple, Optional, List

import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from numpy import ndarray
from tensorflow import Tensor


def darknet53(input_data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    :darknet53: darknet53 model

    :param input_data: input data tensorflow
    :type input_data: Tensor
    :return:
    darknet53 outputs
    """

    input_data = convolutional(input_data, (3, 3, 3, 32))
    input_data = convolutional(input_data, (3, 3, 32, 64),
                                      downsample=True)

    for i in range(1):
        input_data = residual_block(input_data, 64, 32, 64)

    input_data = convolutional(input_data, (3, 3, 64, 128),
                                      downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128, 64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256),
                                      downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512),
                                      downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024),
                                      downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolutional(input_layer: Tensor, filters_shape: Tuple,
                  downsample: bool = False,
                  activate: bool = True,
                  bn: bool = True):
    """
    :convolutional: custom convolution layer

    :param input_layer: input layer
    :type input_layer: Input

    :param filters_shape: filters shape
    :type filters_shape: tuple

    :param downsample: downsample flag
    :type downsample: bool

    :param activate: leaky relu
    :type activate: bbol

    :param bn: batchnorm flag
    :type bn: bool

    :return:
    convolution tensor
    """
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(
            input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides, padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_layer: Tensor, input_channel: Tensor,
                   filter_num1, filter_num2) -> Tensor:
    """
    :residual_block: residual block function
    :param input_layer: input tensor
    :type input_layer: Tensor

    :param input_channel: input tensor
    :type input_channel: Tensor

    :param filter_num1: size of input channels
    :type filter_num1: int

    :param filter_num2: size of output channels
    :type filter_num2: int
    :return:
    output of residual block
    """
    short_cut = input_layer
    conv = convolutional(input_layer,
                         filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))

    residual_output = short_cut + conv
    return residual_output


def upsample(input_layer: Tensor):
    """
    :upsample: upsample function
    :param input_layer: input layer tensor
    :type input_layer: Tensor
    :return:
    resized tensor
    """
    return tf.image.resize(input_layer, (
        input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def image_preporcess(image: ndarray, target_size: int,
                     gt_boxes: Optional[ndarray] = None) -> Tensor:
    """
    :image_preprocess: scale with padding the image to a target size
    :param image: RGB image
    :type image: ndarray
    :param target_size: image target size
    :type target_size: int
    :param gt_boxes: ndarray
    :return:
    """
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def bboxes_iou(boxes1, boxes2) -> float:
    """
    :bboxes_iou: intersection over union between two bboxes.

    :param boxes1: bbox
    :type boxes1: ndarray
    :param boxes2:bbox
    :type boxes2: ndarray

    :return:
    intersection over union score
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
            boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
            boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes: list, iou_threshold: float,
        sigma: float = 0.3, method: str = 'nms'):
    """
    :nms: non-maximum suppression.

    :param bboxes: bbox coordinates.
                   (xmin, ymin, xmax, ymax, score, class)
    :type bboxes: list

    :param iou_threshold: threshold to remove duplicated bbox
    :type iou_threshold: float

    :param sigma: sigma value for softnms
    :param method: nms or soft-nms
    :return:
    ndarray of best bbox

    :Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox: ndarray,
                      org_img_shape: int,
                      input_size: int,
                      score_threshold: float) -> ndarray:
    """
    :postprocess_boxes: post precessing the bboxes.
    :param pred_bbox: bboxes
    :type pred_bbox: ndarray

    :param org_img_shape: original image shape
    :type org_img_shape: int

    :param input_size: image input size
    :type input_size: int

    :param score_threshold: threshold to discard some invalid bboxes
    :type score_threshold: float
    :return:
    """
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5],
                               axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:],
                                           [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                 (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(
        np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def get_bbox_coords(arr: ndarray) -> ndarray:
    """
    :get_bbox_coords: get bboxes coordinates.

    :param arr: bbox of polygons arr
    :type arr: ndarray
    :return:
    """
    x_min, x_max = min(arr[:, 0]), max(arr[:, 0])
    y_min, y_max = min(arr[:, 1]), max(arr[:, 1])

    return np.array([[x_min, y_min], [x_max, y_max]])


def get_object(model: Model, input_image: ndarray,
              input_size: int = 416) -> ndarray:
    """
    :crop_card: run yolov3 to crop the card region.
    :param model: keras Model
    :type model_path: Model
    :param input_image: RGB image
    :type input_image: ndarray

    :param input_size:input size
    :return:
    card crop
    """
    original_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = image_preporcess(np.copy(original_image),
                                  [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    print(pred_bbox)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = postprocess_boxes(pred_bbox, original_image_size,
                               input_size, 0.3)
    # bboxes format bboxes: [x_min, y_min, x_max, y_max, probability, cls_id]
    bboxes = nms(bboxes, 0.1, method='nms')
    # take only the first crop.
    if len(bboxes) != 0:
        bbox_1 = bboxes[0]
        x_min, y_min, x_max, y_max = int(bbox_1[0]), int(bbox_1[1]), int(
            bbox_1[2]), int(bbox_1[3])
        # crop the original image and get the card.
        h, w = original_image.shape[:2]
        crop = original_image[max(0, y_min - 10):min(h, y_max + 10),
               max(0, x_min - 10):min(w, x_max + 10)]
        # save crop
        # cv2.imwrite("crop_id_card.png", crop)
        print('crop done!')
    else:
        crop = None
    return crop


NUM_CLASS = 1
ANCHORS = np.array(
    [1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125,
     3.6875, 7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875
     ]).reshape(3, 3, 2)
STRIDES = np.array([8, 16, 32])
IOU_LOSS_THRESH = 0.5


def YOLOv3(input_layer: Input) -> List:
    """
    Run Yolo3 model.
    :param input_layer: input layer of yolo3
    :type input_layer: Tensor
    :return: list of features maps
    """
    route_1, route_2, conv = darknet53(input_layer)

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv_lobj_branch,
                                      (1, 1, 1024, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv_mobj_branch,
                                      (1, 1, 512, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv_sobj_branch,
                                      (1, 1, 256, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_output, i: int = 0) -> Tensor:
    """
    :decode: Decode the output of Yolo3 to get the bbox.

    :param conv_output: output feature map
    :type conv_output: Tensor

    :param i: anchors index
    :type i: int

    :return: tensor of shape [batch_size, output_size,
     output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(conv_output, (
        batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis],
                [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :],
                [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :],
                      [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1: ndarray, boxes2: ndarray) -> float:
    """
    :bbox_iou: intersection over union between two bboxes.

    :param boxes1: bbox
    :type boxes1: ndarray
    :param boxes2:bbox
    :type boxes2: ndarray

    :return:
    intersection over union score
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    """
    :bbox_giou: Generalized intersection over union between two bboxes.

    :param boxes1: bbox
    :type boxes1: ndarray
    :param boxes2:bbox
    :type boxes2: ndarray

    :return:
    intersection over union score
    """
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
            boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
            boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area + 1e-10
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred: Tensor, conv: Tensor, label: Tensor,
                 bboxes: Tensor, i: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
    """
    :compute_loss: compute loss of yolo3
    :param pred: predict tensor
    :type pred: Tensor
    :param conv: target tensor
    :type conv: Tensor
    :param label: label tensor
    :type label: Tensor
    :param bboxes: bboxes tensor
    :type bboxes: Tensor
    :param i: index
    :type i: int
    :return:
    tuple of losses tensors
    """
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv,
                      (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:,
                                                                :, :, :,
                                                                3:4] / (
                              input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                   bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH,
                                                 tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss


def create_yolo3_model(input_size: int = 416):
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    return model


if __name__ == "__main__":
    model = create_yolo3_model()

    images = np.random.uniform(0, 255., (10, 416, 416, 3))
    images = images.astype(np.float32)
    for idx, image in enumerate(images):
        # here i'll print pred_bbox variable in get_object
        # We remark that the first iteration is good but the rest is NaN
        print(f"Iteration : {idx}")
        get_object(model, input_image=image)