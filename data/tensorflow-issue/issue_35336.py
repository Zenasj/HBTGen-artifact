import random
from tensorflow import keras

# pip install tensorflow-gpu==1.14.0
# pip pandas
#%%
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob

#%%
# input image dimensions
img_h = 224
img_w = 224
channels = 3

# information for dataset
dataset_path = "dataset-imagenet/"
num_classes = 1000
num_testing = 50000

#%%
class DataGenerator:

    def __init__(self, dataframe, batch_size, run_aug = True):

        self.total_len  = len(dataframe.index)
        self.batch_size = batch_size
        self.run_aug = run_aug
        self.dataframe  = dataframe
        self.on_epoch_end()

    def __build_pipeline(self, file_path, labelY):

        # mapping function in tf
        def preprocess_fn(file_path, labelY):

            def fn_x(img_array):

                img_array = img_array.numpy()

                if self.run_aug == 1:
                    # image's range is [0,255]
                    image = img_array

                if self.run_aug >= 2:
                    # image's range is [0,1]
                    image = img_array / 255.0

                if self.run_aug == 3:
                    # std normalization
                    image[0,:,:] -= 0.485
                    image[1,:,:] -= 0.456
                    image[2,:,:] -= 0.406
                    image[0,:,:] /= 0.229
                    image[1,:,:] /= 0.224
                    image[2,:,:] /= 0.225

                return image

            def fn_y(label):
                return tf.keras.utils.to_categorical(label , num_classes)

            # read image from files
            image = tf.io.read_file(file_path)
            image = tf.image.decode_image(image, channels=channels)
            aug_size = 256
            imageX = tf.compat.v1.image.resize_image_with_pad(image, aug_size, aug_size)
            imageX = tf.image.resize_with_crop_or_pad(image, img_h, img_w)

            # do normalizarion
            [imageX] = tf.py_function(fn_x, [imageX], [tf.float32])
            imageX.set_shape([img_h, img_w, channels])
            imageX = tf.image.random_flip_left_right(imageX)

            [labelY] = tf.py_function(fn_y, [labelY], [tf.float32])
            labelY.set_shape([num_classes])

            return imageX, labelY

        dataset = tf.data.Dataset.from_tensor_slices( (file_path, labelY) )
        dataset = dataset.shuffle(batch_size * 8)
        dataset = dataset.repeat()
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.dataset   = dataset

    def  __len__(self):

        return self.total_len // self.batch_size

    def on_epoch_end(self):

        cleanX = np.array(self.dataframe["File"])
        totalY = np.array(self.dataframe["One-hot"])

        # run permutation
        rand_idx = np.random.permutation(self.total_len)
        cleanX = cleanX[rand_idx]
        totalY = totalY[rand_idx]

        self.__build_pipeline(cleanX, totalY)

#%%
def build_clf(model_name):

    if model_name == "ResNet50":
        clf_model = tf.keras.applications.ResNet50(include_top=True, pooling='max', weights='imagenet')

    if model_name == "DenseNet121":
        clf_model = tf.keras.applications.DenseNet121(include_top=True, pooling='max', weights='imagenet')

    if model_name == "MobileNetV2":
        clf_model = tf.keras.applications.MobileNetV2(include_top=True, pooling='max', weights='imagenet')

    if model_name == "InceptionV3":
        clf_model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')


    clf_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return clf_model

#%%
def list_testing_data(classes, file_path, onehot_map):

    try:
        testing_data = pd.read_pickle('imagenet_test_list.pkl')
        print('[Successful] Testing_data loaded from pickle ...')
    except:
        testing_image_info = []
        for iter_class in classes:
            files = glob(os.path.join(file_path, iter_class, '*.JPEG'))
            for iter_img in files:
                data_info = [iter_img, iter_class]
                testing_image_info.append(data_info)

        testing_data = pd.DataFrame(testing_image_info, columns=['File', 'Class'])
        testing_data["One-hot"] = testing_data["Class"].replace(onehot_map, inplace=False)

        testing_data.to_pickle('imagenet_test_list.pkl')

    assert(testing_data.shape[0] == num_testing, "[Fatal] Mismatched total length of testing data")
    return testing_data

#%%
if __name__ == '__main__':

    # set GPU
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Hyperparameters
    batch_size = 100
    epochs = 5

    # load one-hot labels
    file_path = dataset_path + 'val'
    classes = os.listdir(file_path)
    list_class = sorted( list( set(classes) ) )
    onehot_map = dict( zip( list_class, list(range(0, num_classes)) ))

    # load list of validation data, those data should be considered as testing data
    testing_data = list_testing_data(classes, file_path, onehot_map)

    # build data generator
    gen_type1 = DataGenerator(testing_data, batch_size, run_aug=1)
    gen_type2 = DataGenerator(testing_data, batch_size, run_aug=2)
    gen_type3 = DataGenerator(testing_data, batch_size, run_aug=3)
    gen_list = [gen_type1, gen_type2, gen_type3]

    # build model
    model_list = ["ResNet50", "DenseNet121", "MobileNetV2"]
    
    # print result for type1
    test_gen = gen_type1
    for model_name in model_list:
        model = build_clf(model_name)
        meta_string = '[Testing][pixel vales are from (0,255)][model:{:s}] '.format(model_name)
        prefix_string = ''
        output = model.evaluate(test_gen.dataset, steps = test_gen.__len__())
        for ii in range( len( model.metrics_names) ):
            meta_string = meta_string + '- {:s}{:s}: {:.3f} '.format(prefix_string, model.metrics_names[ii], output[ii])

        print(meta_string)

    # print result for type2
    test_gen = gen_type2
    for model_name in model_list:
        model = build_clf(model_name)
        meta_string = '[Testing][pixel vales are from (0,1)][model:{:s}] '.format(model_name)
        prefix_string = ''
        output = model.evaluate(test_gen.dataset, steps = test_gen.__len__())
        for ii in range( len( model.metrics_names) ):
            meta_string = meta_string + '- {:s}{:s}: {:.3f} '.format(prefix_string, model.metrics_names[ii], output[ii])

        print(meta_string)

    # print result for type3
    test_gen = gen_type3
    for model_name in model_list:
        model = build_clf(model_name)
        meta_string = '[Testing][pixel vales are normalized from (-1,1)][model:{:s}] '.format(model_name)
        prefix_string = ''
        output = model.evaluate(test_gen.dataset, steps = test_gen.__len__())
        for ii in range( len( model.metrics_names) ):
            meta_string = meta_string + '- {:s}{:s}: {:.3f} '.format(prefix_string, model.metrics_names[ii], output[ii])

        print(meta_string)

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = image/255
  image = tf.image.resize(image, (224, 224))
  image = image[None, ...]
  return image