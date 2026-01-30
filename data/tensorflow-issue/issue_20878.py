from tensorflow.keras import layers
from tensorflow.keras import models

from keras.models import Sequential, Model
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from keras import optimizers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

train_path='/home/sober/code/jupyter/ftp_code/AndroidTensorflow/tensorflow-for-poets-2/tf_files/flower_photos'
val_path='/home/sober/code/jupyter/ftp_code/AndroidTensorflow/tensorflow-for-poets-2/tf_files/flower_photos'
batch_size = 16
epochs = 200

input = Input(shape = (150, 150, 3), name = 'input')
x = Conv2D(32, (3, 3), activation='relu')(input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(2, activation='sigmoid', name = 'output')(x)
x = Dense(5, activation='softmax', name = 'output')(x)
model = Model(inputs = input, outputs= x)

sgd = optimizers.SGD(lr=0.000005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=1340 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        shuffle=True,
        validation_steps=64 // batch_size)
model.save('./weight/flower5.h5')  # always save your weights after training or during training

# convert
output_names = [node.op.name for node in model.outputs]

export_dir = './weight/'
sess = K.get_session()
save_graph_to_file(sess,  export_dir + "flower5.pb", output_names)

def save_graph_to_file(sess,  graph_file_name, output_names):
    output_graph_def = graph_util.convert_variables_to_constants(
      sess,  sess.graph.as_graph_def(),  output_names)
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

import numpy as np
import tensorflow as tf
from PIL import Image
import os
# filepath='./weight/flower_gpu.tflite'
filepath='./weight/flower5.lite'
val_path = '/home/sober/ftp_bk/Dataset/Tmp_Train/flo/'
shape=(150, 150)
# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path=filepath)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
# change the following line to feed into your own data.

for d in os.listdir(val_path):
    dirpath = val_path+ d + '/'
    for f in os.listdir(dirpath):
        file = dirpath + f
        print(file)
        im = Image.open(file)
        im = im.resize(shape, Image.ANTIALIAS)
        im = np.asarray(im)
        im = im[np.newaxis, :]
        im = np.asarray(im)
#         print(np.array2string(model.predict(np.asarray(im)), formatter={'float_kind':lambda x: "%.2f" % x}))
        input_data = np.array(im, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
#         output_data.sort()
        print(np.array2string(output_data, formatter={'float_kind':lambda x: "%.2f" % x}))
    
"daisy dandelion roses sunflowers tulips"