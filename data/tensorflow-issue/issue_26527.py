import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def load_image_into_numpy_array(image_path):
  image = Image.open(image_path)
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


if __name__ == "__main__":
    TEST_IMAGE_PATHS = 'train'
    output = {}
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu, input_shape=(360, 640,  3)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    labels = {}
    print("Loaded labels")
    with open("labels.json") as label_f:
        labels = json.loads(''.join(label_f.readlines()))
    train_images = []
    train_labels = []
    count = 0
    for (dirpath, dirnames, filenames) in os.walk(TEST_IMAGE_PATHS):
        for filename in filenames:
            if filename in labels:
                img = load_image_into_numpy_array(os.path.join(dirpath, filename))
                train_images.append(img)
                train_labels.append([labels[filename]])
        else:
            continue
        break
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    model.fit(train_images, train_labels, epochs=5, batch_size=396)