import numpy as np
import tensorflow as tf
from tensorflow import keras

class BasicDatasetLight(object):
    def __init__(self, run_data, batch_size):
        """
        :param run_data: a dictionary storing the data info
        :param batch_size:
        """
        from os.path import splitext
        data_array = np.load(run_data["data_file"], allow_pickle=True)
        self.run_info = run_data
        self.x, self.y = data_array[:,:-1], data_array[:,-1].astype(dtype=int)
        self.img_num = len(data_array[0][0])        # input image number
        _, self.image_type = splitext(data_array[0][0][0])      # input image type
        self.model_input_num = run_data["input_num"]
        self.gen_out_type = self.__gen_out_type_shape()[0]     # get output type tuple
        self.gen_out_shape = self.__gen_out_type_shape()[1]     # get output shape tuple
        self.batch_size = batch_size

    def __len__(self):
        """
        generate batch number
        :return:
        """
        return len(self.x) // self.batch_size

    def __gen_out_type_shape(self):
        """
        generate output type tuple according the input
        :return:
        """
        peep_file = self.x[:,0][0][0]
        peep_shape = self.run_info["image_shape"]
        peep_image = _tf_load_img(peep_file, peep_shape[0], decode_image=self.image_type)
        out_type = peep_image.dtype
        # type list [type] * image_input_num + [audio type]
        type_list = [out_type] * (len(peep_shape)*self.img_num) + [out_type]
        # shape list [shape] * image_input_num + [audio shape]
        shape_list = [peep_shape[0]]*(len(peep_shape)*self.img_num) + [self.run_info["audio_shape"]]
        gen_out_shape = fake_gen_shape(shape_list)
        assert len(type_list)==self.model_input_num,"wrong input number"
        return (tuple(type_list),tf.int8), (gen_out_shape, tf.TensorShape(1,))

    def give_dataset(self):
        """
        serve the dataset to model
        :return:
        """
        ds = tf.data.Dataset.from_generator(
            lambda: _get_generator_light(self.x, self.y, self.run_info,self.image_type),
             output_types=self.gen_out_type,
             output_shapes=self.gen_out_shape)
        ds = ds.repeat()
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

def _tf_load_img(img_file, target_shape, decode_image=".jpg"):
    """
    tensorflow load image
    :param img_file:
    :param target_shape: (H x W x C) channel last
    :param decode_image: image extension for decoding mode
    :return:
    """
    image = tf.io.read_file(img_file)
    # image shape channel last
    if "png" in decode_image: image = tf.image.decode_png(image, channels=target_shape[-1])
    else: image = tf.image.decode_jpeg(image, channels=target_shape[-1])
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (target_shape[0], target_shape[1]))
    return image


def _get_generator_light(x, y, run_info, image_type=".jpg"):
    """
    a generator function for tensorflow dataset
    :param x:   input data
    :param y:   label
    :param run_info:    run information dictionary
    :param image_type:  image type for identifying decoding mode
    :return: ([input list], output)
    """
    # images number x copies (models may need different inputs) + audio input
    image_in,audio_in,y = x[:,0],x[:,1],y
    image_shape = run_info["image_shape"]
    image_num = len(image_in[0])
    decode_image = image_type
    buff_size = len(image_shape)*image_num + 1
    for data_read_index in range(len(x)):
        # data_buff = np.empty(buff_size, dtype=object)
        data_buff = [None] * buff_size
        buff_index = 0
        for ii in range(image_num):
            for neti in range(len(image_shape)):
                image = _tf_load_img(image_in[data_read_index][ii],
                                     image_shape[neti],
                                     decode_image=decode_image)
                data_buff[buff_index] = image
                buff_index += 1
        data_buff[buff_index] = np.expand_dims(audio_in[data_read_index],axis=0)
        label = y[data_read_index]
        assert buff_index==buff_size-1, "wrong data buff size"
        yield data_buff, label

class BasicGeneratorLight(tf.keras.utils.Sequence):
    def __init__(self, run_data, batch_size):
        """
        :param run_data: a dictionary storing the data info
        :param batch_size:
        """
        from os.path import splitext
        data_array = np.load(run_data["data_file"], allow_pickle=True)
        self.run_info = run_data
        self.x, self.y = data_array[:,:-1], data_array[:,-1].astype(dtype=int)
        self.img_num = len(data_array[0][0])        # input image number
        _, self.image_type = splitext(data_array[0][0][0])      # input image type
        self.model_input_num = run_data["input_num"]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        img_num = self.img_num    # check how many images we have
        imagenet_shapes = self.run_info["image_shape"]      #  input shape for each image net (ORDER MATTERS!)
        imgnet_num = len(imagenet_shapes)
        input_num = img_num*imgnet_num+1
        model_data = np.empty(input_num,dtype=object)     # data buffer
        model_data_index = 0
        audio_buff = np.empty([self.batch_size] + list(self.run_info["audio_shape"]), dtype=float)
        audio_read = True
        for ii in range(img_num):   # read every image
            for neti in range(imgnet_num):  # prepare for each image model
                image_buff = np.empty([self.batch_size] + list(imagenet_shapes[neti]), dtype=float)
                # load data per img
                for di in range(self.batch_size):
                    curr_imgfile = batch_x[di, 0][ii]
                    #image_buff[di, :, :, :] = _tf_load_img(curr_imgfile,
                    #                                       imagenet_shapes[neti], self.image_type)
                    image_buff[di, :, :, :] = _keras_load_img(curr_imgfile,
                                                              imagenet_shapes[neti],'vgg')  # use 'vgg' for all
                    if audio_read:
                        aud_len = batch_x[di, 1].shape[-1]
                        audio_buff[di, 0, :, :aud_len] = batch_x[di, 1]  # set channle axis to 0 as only mono channel
                audio_read = False  # disable audio_read after reading audio one time
                model_data[model_data_index] = image_buff
                model_data_index += 1
        model_data[model_data_index] = audio_buff  # append audio data to the end
        assert model_data_index==input_num-1, "input number wrong"
        return list(model_data), batch_y