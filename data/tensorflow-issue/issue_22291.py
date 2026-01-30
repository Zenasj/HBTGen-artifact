from tensorflow import keras

import tensorflow as tf
tf.enable_eager_execution()

# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt
#import inception_v4 
# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

annotation_zip = tf.keras.utils.get_file('captions.zip', 
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

name_of_zip = 'train2014.zip'
if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
    image_zip = tf.keras.utils.get_file(name_of_zip, 
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
    PATH = os.path.dirname(image_zip)+'/train2014/'
else:
    PATH = os.path.abspath('.')+'/train2014/'


# read the json file# read  
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# storing the captions and the image name in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
    
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# selecting the first 30000 captions from the shuffled set
num_examples = 30000

def load_image(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(x = img)
    return img, image_path

image_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

startTime=time.time()
# getting the unique images
encode_train = sorted(set(img_name_vector))

# feel free to change the batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(
                                encode_train).map(load_image).batch(16)

for img, path in image_dataset:
    batch_features_0 = image_features_extract_model(img)
  
    batch_features = tf.reshape(batch_features_0, (batch_features_0.shape[0], -1, batch_features_0.shape[3]))
    Nan = np.any(np.isnan(batch_features))

    for bf, p in zip(batch_features, path):

        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

### test image
image = './COCO_val2014_000000000042.jpg'
print(load_image(image))

import tensorflow as tf
tf.enable_eager_execution()
def load_image(image_path):
  img = tf.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize_images(img, (224, 224))
  img = tf.keras.applications.vgg16.preprocess_input(x = img)
  return img, image_path

load_image('~/Downloads/coco_example.png')