import os
import glob
import boto3
import numpy as np
import tensorflow as tf
import PIL
import matplotlib
import sys
from distutils.version import StrictVersion
from PIL import Image
from utils import ops as utils_ops
from matplotlib import pyplot as plt

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
import json

# Download Model From S3.
MODEL_GRAPH_DEF_PATH = '/tmp/model.pb'
BUCKET_NAME = 'solver'

def lambda_handler(event, context):
  s3 = boto3.resource('s3')
  print(event['queryStringParameters'])
  img_url = 'images/' + event['queryStringParameters']['image_name']
  model_path = 'models/' + event['queryStringParameters']['model_name'] + '.pb'
  print("DOWNLOADING MODEL " + model_path)
  s3.Bucket(BUCKET_NAME).download_file(model_path,MODEL_GRAPH_DEF_PATH)
  print("DOWNLOADING IMAGE " + img_url)
  s3.Bucket(BUCKET_NAME).download_file(img_url,'/tmp/image.jpg')
  label_path = 'models/' + event['queryStringParameters']['model_name'] + '.pbtxt'
  print("DOWNLOADING LABEL MAP " + label_path)
  s3.Bucket(BUCKET_NAME).download_file(label_path,'/tmp/label_map.pbtxt')
  print(glob.glob("/tmp/*"))
  detect()

def detect():
  PATH_TO_CKPT = '/tmp/model.pb'
  PATH_TO_IMAGE = '/tmp//image.jpg'
  NUM_CLASSES = 1
  label_map = label_map_util.load_labelmap('/tmp/label_map.pbtxt')
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

  # Define input and output tensors (i.e. data) for the object detection classifier

  # Input tensor is the image
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

  # Output tensors are the detection boxes, scores, and classes
  # Each box represents a part of the image where a particular object was detected
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

  # Each score represents level of confidence for each of the objects.
  # The score is shown on the result image, together with the class label.
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

  # Number of objects detected
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
  print("SOMETHING WAS LOADED")
  print(num_detections)