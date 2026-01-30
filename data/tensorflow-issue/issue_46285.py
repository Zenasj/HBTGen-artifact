import random

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt

import cv2
import time
import numpy as np

from PIL import Image

IMAGE_PATH = "C:/Users/Reno/Documents/TensorFlow/workspace/training_walnoot/evaluate/27.jpg"
MODEL_PATH = "C:/Users/Reno/Documents/TensorFlow/workspace/training_walnoot/exported-models/walnoot_model/saved_model/saved_model.tflite"

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  
  print(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results
  
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
_, IMG_HEIGHT, IMG_WIDTH, _ = interpreter.get_input_details()[0]['shape']

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    original_image = img
    resized_img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image
    
LABEL_DICT = {0: 'Walnoot'
}

COLORS = np.random.randint(0, 255, size=(len(LABEL_DICT), 3), dtype="uint8")

def display_results(image_path, threshold=0.3):
    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(image_path)
    # print(preprocessed_image.shape, original_image.shape)

    # =============Perform inference=====================
    start_time = time.monotonic()
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
    print(f"Elapsed time: {(time.monotonic() - start_time)*1000} miliseconds")

    # =============Display the results====================
    original_numpy = original_image.numpy()
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_numpy.shape[1])
        xmax = int(xmax * original_numpy.shape[1])
        ymin = int(ymin * original_numpy.shape[0])
        ymax = int(ymax * original_numpy.shape[0])

        # Grab the class index for the current iteration
        idx = int(obj['class_id'])
        # Skip the background
        if idx >= len(LABEL_DICT):
            continue
        
        # draw the bounding box and label on the image
        color = [int(c) for c in COLORS[idx]]
        cv2.rectangle(original_numpy, (xmin, ymin), (xmax, ymax), 
                    color, 2)
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.2f}%".format(LABEL_DICT[obj['class_id']],
            obj['score'] * 100)
        cv2.putText(original_numpy, label, (xmin, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # return the final ima
    original_int = (original_numpy * 255).astype(np.uint8)
    return original_int
    
resultant_image = display_results(IMAGE_PATH)
img = Image.fromarray(resultant_image)

plt.figure()
plt.imshow(img)

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

plt.show()

print(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))

[100.]

import tensorflow as tf

saved_model_dir = "C:/Users/xxxxx/Documents/TensorFlow/workspace/training/exported-models/model_lite/saved_model/"

# Convert the model to TF lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.experimental_new_converter=True
tflite_model = converter.convert()

# Serialize the model
open(saved_model_dir + 'saved_model.tflite', 'wb').write(tflite_model)