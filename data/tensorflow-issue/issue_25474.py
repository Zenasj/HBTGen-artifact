import random

cc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
output = cv2.VideoWriter('output.avi', cc, 8, (640, 480))
while cv2.VideoCapture(0).isOpened:
      flag, frame = cv2.VideoCapture(0).read()
      if not flag:
         break
      img = frame[:, :, ::-1]
      prediction = model.detect([img], verbose=0)

import os
import cv2
import sys
import numpy as np
import mrcnn.utils
import mrcnn.config
from pathlib import Path
from mrcnn.model import MaskRCNN
from mrcnn import visualize
from keras.utils import plot_model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# COCO Class names
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

detect_names = [
    'backpack', 'umbrella', 'handbag',
    'sports ball', 'bottle', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'donut', 'cake', 'potted plant',
    'mouse', 'keyboard', 'cell phone', 'book', 'clock',
    'vase', 'scissors', 'toothbrush'
]


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Root directory of the project
ROOT_DIR = Path(".")
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
# import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Video file or camera to process - set this to 0 to use your webcam instead of a video file
VIDEO_SOURCE = 0

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

plot_model(model.keras_model, to_file='model_concat.png', expand_nested=False)

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# saving video output
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
output_video = cv2.VideoWriter('output.avi', fourcc, 8, (640, 480))

# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]
    #rgb_image = np.random.randn(480, 640, 3)

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    colors = visualize.random_colors(r['rois'].shape[0])
    for bbox, mask, cls, score, color in zip(
            r['rois'], r['masks'], r['class_ids'], r['scores'], colors):
        y1, x1, y2, x2 = bbox
      
        if class_names[cls] in detect_names:
            text = class_names[cls] + " {:.2f}%".format(score * 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          tuple(map(lambda x: int(x * 255), color)), 1)
            cv2.putText(frame, text, (10, y2), cv2.FONT_HERSHEY_DUPLEX, 3.0,
                        (0, 255, 0), 2, cv2.FILLED)

            if success is True:
                output_video.write(frame)

        cv2.imshow('Video', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
output_video.release()
cv2.destroyAllWindows()