import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import cv2
import os
import config
import random
import csv
import time

from tensorflow.python.lib.io import _pywrap_record_io

"""
gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0],
   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=128)])
   
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

tf.config.experimental.per_process_gpu_memory_fraction = 0.9
tf.config.experimental.per_process_memory_growth = True
"""

#gpus = tf.config.experimental.list_physical_devices('CPU')

#tf.config.experimental.set_virtual_device_configuration(gpus[0],
#   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


class Autoencoder(tf.keras.Model):
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        """
        self.encoder_a = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(448, 448, 1)),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1, 1, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1, 2, 2, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1, 1, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1, 2, 2, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1, 1, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1, 1, 1), padding="same", activation="tanh"),
            #tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh"),
            #tf.keras.layers.Dense(512, activation="sigmoid")
        ])
        
        self.encoder_b = tf.keras.Sequential([
            #tf.keras.layers.Input(shape=(448, 448, 1)),
            #input_shape=(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 2, 2, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1, 1, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 2, 2, 1), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1, 1, 1), padding="same", activation="tanh"),
            tf.keras.layers.Dense(1024, activation="swish"),
            tf.keras.layers.Dense(512, activation="swish"),
            tf.keras.layers.Dense(256, activation="swish")
        ])
        """
        self.encoder_a = tf.keras.Sequential([
            #tf.keras.layers.InputLayer(input_shape=(1), batch_size=2),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )),
            #tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh"),
            #tf.keras.layers.Dense(512, activation="sigmoid")
        ])
        
        self.encoder_b = tf.keras.Sequential([
            #tf.keras.layers.Input(shape=(448, 448, 1)),
            #input_shape=(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Dense(1024, activation="swish",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.04),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Dense(512, activation="swish",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.04),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                ),
            ), #, activation="relu"
            tf.keras.layers.Dense(256, activation="swish",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.04),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                ),
            ) #, activation="relu" #swish?
        ])
        
        self.decoder = tf.keras.Sequential([
            #input_shape=(),
            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
            tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=(3,3), strides=(1, 1), padding="same", activation="sigmoid",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1
                )
            ),
        ])
        
    def set_initial_values(self):
        for layer in self.encoder_a.layers:
            #print(np.layer.get_weights().shape)
            print("layer weights")
            for weight in layer.get_weights():
                print(weight)
            #weights = tf.random.normal(shape=(), mean=0.1, stddev=0.1)
            #layer.set_weights(weights)
        """
        for layer in self.encoder_b.layers:
            print(layer.get_weights().shape)
            #weights = tf.random.normal(shape=(), mean=0.1, stddev=0.1)
            #layer.set_weights(weights)
            
        for layer in self.decoder.layers:
            print(layer.get_weights().shape)
            #weights = tf.random.normal(shape=(), mean=0.1, stddev=0.1)
            #layer.set_weights(weights)
        """
    
    def call(self, x):
        encoded_a = self.encoder_a(x)
        print("shape a")
        print(encoded_a.get_shape())
        encoded_b = self.encoder_b(encoded_a)
        print("shape b")
        print(encoded_b.get_shape())
        #print(encoded_a.shape)
        fusion = self.fusion(encoded_a, encoded_b)
        print(fusion.get_shape())
        decoded = self.decoder(fusion)
        print(decoded.get_shape())
        #decoded = tf.image.resize(decoded, [448, 448], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return decoded
        
    def fusion_alt(self, mid_features, global_features):
        print("fusion shapes")
        print(global_features.get_shape())
        print(mid_features.get_shape())
        mid_features_shape = mid_features.get_shape().as_list()
        mid_features_reshaped = tf.reshape(mid_features, [1, mid_features_shape[1]*mid_features_shape[2], 256])
        fusion_level = []
        for j in range(mid_features_reshaped.shape[0]):
            for i in range(mid_features_reshaped.shape[1]):
                see_mid = mid_features_reshaped[j, i, :]
                see_mid_shape = see_mid.get_shape().as_list()
                see_mid = tf.reshape(see_mid, [1, see_mid_shape[0]])
                global_features_shape = global_features[j, :].get_shape().as_list()
                see_global = tf.reshape(global_features[j, :], [1, global_features_shape[0]])
                fusion = tf.concat([see_mid, see_global], 1)
                fusion_level.append(fusion)
        fusion_level = tf.stack(fusion_level, 1)
        #fusion_level = tf.reshape(fusion_level, [config.BATCH_SIZE, 28, 28, 512])
        #fusion_level = tf.reshape(fusion_level, [-1, 28, 28, 512])
                
        #fusion_level = tf.reshape(fusion_level, [4, 28, 28, 512])
        #fusion_level = tf.reshape(fusion_level, [config.BATCH_SIZE, 28, 28, 2048])
        fusion_level = tf.reshape(fusion_level, [1, mid_features_shape[1], mid_features_shape[2], 512])
        return fusion_level
    
    def fusion(self, a, b):
        """
        a_shape = a.get_shape().as_list()
        print(a_shape)
        shapes = [a_shape[1], a_shape[2]]
        #for shape_value in a_shape:
        #    if(shape_value != 512):
        #        shapes.append(shape_value)
        last_shape = (256/(shapes[0]*shapes[1]))
        print("last shape")
        print(last_shape)
        last_shape = int(last_shape)
        tf.reshape(b, (shapes[0], shapes[1], last_shape))
        #b.reshape(shapes[0], shapes[1], last_shape)
        #return a, b
        #self.fusion_layer = tf.keras.Sequential([
        """
        return tf.keras.layers.Concatenate(axis=3)([a, b])
        #])
        #return self.fusion_layer
        
    
class DATA():
    def __init__(self):
        self.test = 0
    
    def write_tf_record(self, dirname):
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        filelist = os.listdir(self.dir_path)
        filelist = self.listdir_nohidden(filelist)
        random.shuffle(filelist)
        record_file = '/Users/siggi/VideoColor/videocolor/DATASET/TV/TFRECORD/images.tfrecords'
        with tf.io.TFRecordWriter(record_file) as writer:
            for filename in filelist:
                filename = os.path.join(dirname, filename)
                print(filename)
                img = cv2.imread(filename, 1)
                #img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                labimg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                gray_image = np.reshape(labimg[:,:,0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1))
                color_image = labimg[:, :, 1:]
                #example = tf.train.Example(
                #   features=tf.train.Features(
                #
                #    )
                #)
                example_proto = {
                    "image_l": self._float_feature(gray_image.flatten()),
                    "image_ab": self._float_feature(color_image.flatten())
                }
                #print(example_proto)
                #feature_description = {
                    #"image_name": self._bytes_feature(img_file),
                #    "image_l": tf.io.FixedLenFeature([], tf.float32),
                #    "image_ab": tf.io.FixedLenFeature([], tf.float32),
                    #"image_embedding": self._float32_list(img_embedding.flatten()),
                #}
                
                example_proto = tf.train.Example(features=tf.train.Features(feature=example_proto))
                writer.write(example_proto.SerializeToString())

                #example = tf.io.parse_single_example(example_proto, feature_description)

                #self.write(example.SerializeToString())
                #image_string = open(filename, 'rb').read()
                #tf_example = image_example(image_string, label)
                #writer.write(example.SerializeToString())
        
            
    def read_tf_record_init(self):
        self.offset = 0
        record_file = '/Users/siggi/VideoColor/videocolor/DATASET/TV/TFRECORD/images.tfrecords'
        self.raw_data = _pywrap_record_io.RandomRecordReader(record_file)
        
    def index_tf_record(self):
        self.read_tf_record_init()
        with open('/Users/siggi/VideoColor/videocolor/DATASET/TV/TFRECORD/index.txt', 'w') as f:
            do_loop = True
            while(do_loop):
                try:
                    example, offset = self.raw_data.read(self.offset)
                    self.offset = offset
                    f.write(str(offset)+',')
                except:
                    print("end of file")
                    do_loop = False
    
    def read_indicies(self):
        with open('/Users/siggi/VideoColor/videocolor/DATASET/TV/TFRECORD/index.txt', mode='r') as file:
            csv_index = list(csv.reader(file))[0]
            random.shuffle(csv_index)
            self.csv_index = csv_index
            self.csv_index_offset = 0
            #print(csv_index)
            return csv_index
        
    def get_random_images(self):
        images = []
        i = 0
        while(i < 1):
            index = i+self.csv_index_offset
            offset = self.csv_index[index]
            if(offset == ''):
                index = i+self.csv_index_offset+1
                offset = self.csv_index[index]
            self.offset = int(offset)
            images.append(self.read_tf_record_alt_2())
            i = i + 1
        self.csv_index_offset = self.csv_index_offset+1
        return images
        
        
        

    def read_tf_record_alt_2(self):
        feature_description = {
            #"image_name": self._bytes_feature(img_file),
            "image_l": tf.io.FixedLenFeature([448*448], tf.float32),
            "image_ab": tf.io.FixedLenFeature([448*448*2], tf.float32),
            #"image_embedding": self._float32_list(img_embedding.flatten()),
        }
        
        def _parse_function(example_proto):
            feature = tf.io.parse_single_example(example_proto, feature_description)
            
            image_l = feature["image_l"]
            image_ab = feature["image_ab"]
            
            image_l = image_l.numpy().reshape(448, 448, 1)
            image_ab = image_ab.numpy().reshape(448, 448, 2)
            
            #image_l = image_l.eval(session=tf.compat.v1.Session()).reshape(448, 448, 1)
            #image_ab = image_ab.eval(session=tf.compat.v1.Session()).reshape(448, 448, 2)
            
            
            image_l = np.asarray(image_l)/255
            image_ab = np.asarray(image_ab)/255
            
            return image_l, image_ab
            
        parsed_dataset = []
        
                
        raw_dataset = []
        counter = 0
        offset = self.offset #200
        interval = 1 #20
        add_count = 0
        interval_counter = 1
        
        #self.offset = self.offset + 1
        
        print(self.offset)
        example, offset = self.raw_data.read(self.offset)
        
        self.offset = offset
        
        return _parse_function(example)
        

    def read_img(self, filename):
        print(filename)
        img = cv2.imread(filename, 1)
        height, width, channels = img.shape
        labimg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) #cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)
        l, ab = np.reshape(labimg[:,:,0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)), labimg[:, :, 1:]
        l = np.asarray(l)/255
        ab = np.asarray(ab)/255
        return l, ab
        
    def deprocess(self, imgs):
        imgs = imgs * 255
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        return imgs.astype(np.uint8)
        
    def print_image(self, gray_image, color_image):
        color_unaltered = color_image
        whole_image = np.concatenate((gray_image, color_image), axis=2)
        color_image = self.deprocess(color_image)
        gray_image = self.deprocess(gray_image)
        print(gray_image)
        print(color_image)
        result = np.concatenate((gray_image, color_image), axis=2)
        #whole_image = result
        print("batch shapes")
        print(result.shape)
        #result = np.concatenate((batchX, predictedY), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR,  "_reconstructed.png")
        print(save_path)
        cv2.imwrite(save_path, result)
        print("written")
        zeros = np.ones(shape=(448, 448, 1), dtype=np.uint8)*127
        #zeros = zeros / 2
        result = np.concatenate((zeros, color_image), axis=2)
        print(result.shape)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, "_reconstructed_just_colors.png")
        print(save_path)
        cv2.imwrite(save_path, result)
        print("written")
        zeros = np.ones(shape=(448, 448, 2), dtype=np.uint8)*127
        #zeros = zeros / 2
        result = np.concatenate((gray_image, zeros), axis=2)
        print(result.shape)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, "_reconstructed_just_grayscale.png")
        print(save_path)
        cv2.imwrite(save_path, result)
        
        random_values = np.random.rand(448, 448, 2)
        #random_values = self.deprocess(random_values)
        
        loss = tf.reduce_mean(input_tensor=tf.math.squared_difference(random_values, color_unaltered)).numpy
        #loss = loss.eval(session=tf.compat.v1.Session())
        print("random loss: ")
        print(loss)
        
    def reconstruct(self, batchX, predictedY, filelist):
        #for i in range(config.BATCH_SIZE):
        batchX = self.deprocess(batchX)
        predictedY = self.deprocess(predictedY)
        print("batch shapes")
        print(batchX.shape)
        print(predictedY.shape)
        print(predictedY)
        result = np.concatenate((batchX, predictedY), axis=2)
        #result = np.concatenate((batchX, predictedY), axis=2)
        print(result.shape)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, "reconstructed.png")
        print(save_path)
        cv2.imwrite(save_path, result)
        print("written")
        zeros = np.ones(shape=(448, 448, 1), dtype=np.uint8)*127
        #zeros = zeros / 2
        result = np.concatenate((zeros, predictedY), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, "reconstructed_just_colors.png")
        print(save_path)
        cv2.imwrite(save_path, result)
    
model_a = Autoencoder()
#model_a.build()
model_a.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError())


#checkpoint_path = "cp/variables/variables"
#checkpoint_dir = os.path.join(config.MODEL_DIR, checkpoint_path) #os.path.dirname(checkpoint_path)

checkpoint_path = "checkpoint"
checkpoint_dir = os.path.join(config.MODEL_DIR, checkpoint_path) #os.path.dirname(checkpoint_path)

data = DATA()
#data.index_tf_record()

data.read_tf_record_init()
data.read_indicies()

model_a.load_weights(checkpoint_dir)


counter = 0
interlope = 0
while(counter < 20000):

    images = data.get_random_images()


    values = []
    output = []

    #print(images)


    for l, ab in images:
        values.append(l)
        output.append(ab)


    values = np.array(values)
    output = np.array(output)

    print(values.shape)
    print(output.shape)

    """
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq=4
                                                     )
    """
    #
    if interlope == 0:
        with tf.device("/device:GPU:0"):
            model_a.fit(values, output, batch_size=1, epochs=1, shuffle=False)
    else:
        with tf.device("/device:CPU:0"):
            model_a.fit(values, output, batch_size=1, epochs=1, shuffle=False)
            
    print("fit complete")
    counter = counter + 1
    time.sleep(5)
    
    if interlope == 0:
        interlope = 1
    else:
        interlope = 0
        
    if counter % 10 == 0:
        model_a.save_weights(checkpoint_dir)
        #model_a.save(checkpoint_dir)
        print("saved")
        


"""
l, ab = data.read_img("/Users/siggi/VideoColor/videocolor/DATASET/TV/TRAIN/File 7792.png")

#data.print_image(l, ab)


values = np.array([l])
output = np.array([ab])

print(values.shape)
print(output.shape)
#return
        
#values_2 = np.random.rand(448, 448, 1)



# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=16
                                                 )

load = False

#output_ = np.random.rand(448, 1)
with tf.device("/device:CPU:0"):
    if load:
        model_a.load_weights(checkpoint_dir) #change to
    #else:
        #model_a.set_initial_values()
    model_a.fit(values, output, epochs=6400, shuffle=False, callbacks=[cp_callback])
"""

"""
l, ab = data.read_img("/Users/siggi/VideoColor/videocolor/DATASET/TV/TRAIN/File 7792.png")

values = np.array([l])
output = np.array([ab])
    
with tf.device("/device:CPU:0"):
    model_a.load_weights(checkpoint_dir) #change to
    ab_arr = model_a.predict(values)


data.reconstruct(values[0,:,:,:], ab_arr[0,:,:,:], [1])
"""





