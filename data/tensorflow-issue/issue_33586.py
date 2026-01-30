import random

import tensorflow as tf
import numpy as np
import os

import time
from tensorflow.python.ops import gen_image_ops


def yolo_non_max_suppression(scores, boxes, classes, sess, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
   
    init_val_np = np.array ( [max_boxes], dtype=np.int32) 
    max_boxes_tensor = tf.Variable(max_boxes,  dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    sess.run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (~ 1 line)
    with tf.device("gpu:0"):
        boxes_np = boxes.eval() 
        time0 = time.time()
        score_threshold = 0.1
        soft_nms_sigma=0.0
        pad_to_max_output_size=False
        for i in range(1,2):
            nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
            #nms_indices = nms(boxes_np, 0.5)
            #nms_indices = gen_image_ops.non_max_suppression_v2(boxes, scores, max_boxes_tensor, iou_threshold, )
            #nms_indices = gen_image_ops.non_max_suppression_v3(boxes, scores, max_boxes_tensor, iou_threshold, score_threshold)
            #nms_indices = gen_image_ops.non_max_suppression_v4(boxes, scores, max_boxes_tensor, iou_threshold, score_threshold, pad_to_max_output_size)
            #nms_indices = gen_image_ops.non_max_suppression_v5(boxes, scores, max_boxes_tensor, iou_threshold, score_threshold, soft_nms_sigma)
            if i%100 == 0:
                print (i, (time.time() - time0)/i )
        ### END CODE HERE ###

        ### START CODE HERE ### (~ 3 lines)
        scores = tf.gather(scores, nms_indices)
        boxes = tf.gather(boxes, nms_indices)
        classes = tf.gather(classes, nms_indices)
        summary_writer = tf.summary.FileWriter(os.getenv('TENSORBOARD_DIR'), sess.graph)
        ### END CODE HERE ###

    return scores, boxes, classes

def test_yolo_non_max_suppression():
    with tf.device("cpu:0"):
        with tf.Session() as test_b:
            scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
            boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
            classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
            scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, test_b)
            #scores, boxes, classes = yolo_non_max_suppression0(scores, boxes, classes)
            print("scores[2] = " + str(scores[2].eval()))
            print("boxes[2] = " + str(boxes[2].eval()))
            print("classes[2] = " + str(classes[2].eval()))
            print("scores.shape = " + str(scores.eval().shape))
            print("boxes.shape = " + str(boxes.eval().shape))
            print("classes.shape = " + str(classes.eval().shape))


test_yolo_non_max_suppression()

scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)

scores = tf.random_normal([54,], mean=1, dtype = tf.float32, stddev=4, seed = 1)
boxes = tf.random_normal([54, 4], mean=1, dtype = tf.float32, stddev=4, seed = 1)
classes = tf.random_normal([54,], mean=1, dtype = tf.float32, stddev=4, seed = 1)

py
"""Minimum reproducing example.

Device specs:
    GPU: NVIDIA RTX 2080
    TF version: 1.15 (tensorflow/tensorflow:1.15.0-gpu-py3 container)
    Python version: 3.6.8
    Host driver: 418.56
    CUDA version: 10.1
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_image_ops

# Use placeholder to guarantee that we are feeding floats
boxes = tf.placeholder(tf.float32, [5, 4])
scores = tf.placeholder(tf.float32, [5])
with tf.device('gpu:0'):
    # Try out all 5 of the NMS options, tf.image.non_max_suppression uses V3
    # behind the scenes.
    indices1 = tf.image.non_max_suppression(
        boxes=boxes, scores=scores, max_output_size=2, iou_threshold=0.5)
    indices2 = gen_image_ops.non_max_suppression_v2(
        boxes=boxes, scores=scores, max_output_size=2, iou_threshold=0.5)
    indices3 = gen_image_ops.non_max_suppression_v3(
        boxes=boxes, scores=scores, max_output_size=2, iou_threshold=0.5,
        score_threshold=0.5)
    indices4 = gen_image_ops.non_max_suppression_v4(
        boxes=boxes, scores=scores, max_output_size=2, iou_threshold=0.5,
        score_threshold=0.5, pad_to_max_output_size=False)
    # V5 doesn't exist in TF1.15 yet
    #indices5 = gen_image_ops.non_max_suppression_v5(
    #    boxes=boxes, scores=scores, max_output_size=2, iou_threshold=0.5,
    #    score_threshold=0.5, soft_nms_sigma=0.0, pad_to_max_output_size=False)

# Log device placement to see if NMS op is placed on GPU
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# None of these 5 work
try:
    sess.run(indices1, feed_dict={
        boxes: np.random.random([5, 4]),
        scores: np.random.random([5])})
except Exception as e:
    print('tf.image.non_max_suppression failed: ', e)

try:
    sess.run(indices2, feed_dict={
        boxes: np.random.random([5, 4]),
        scores: np.random.random([5])})
except Exception as e:
    print('gen_image_ops.non_max_suppression_v2 failed: ', e)

try:
    sess.run(indices3, feed_dict={
        boxes: np.random.random([5, 4]),
        scores: np.random.random([5])})
except Exception as e:
    print('gen_image_ops.non_max_suppression_v3 failed: ', e)

try:
    sess.run(indices4, feed_dict={
        boxes: np.random.random([5, 4]),
        scores: np.random.random([5])})
except Exception as e:
    print('gen_image_ops.non_max_suppression_v4 failed: ', e)

#try:
#    sess.run(indices5, feed_dict={
#        boxes: np.random.random([5, 4]),
#        scores: np.random.random([5])})
#except Exception as e:
#    print('gen_image_ops.non_max_suppression_v5 failed: ', e)