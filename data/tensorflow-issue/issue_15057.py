# -*- coding: utf-8 -*-
# @Time    : 2017/10/26 14:09
# @Author  : zhoujun

import tensorflow as tf
from scipy.misc import imread
import time
import os

class PredictionModel:
    
    def __init__(self, model_dir, session=None):
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        start = time.time()
        self.model = tf.saved_model.loader.load(self.session, ['serve'], model_dir)
        print('load_model_time:', time.time() - start)

        self._input_dict, self._output_dict = _signature_def_to_tensors(self.model.signature_def['predictions'])

    def predict(self, image):
        output = self._output_dict
        # 运行predict  op
        start = time.time()
        result = self.session.run(output, feed_dict={self._input_dict['images']: image})
        print('predict_time:',time.time()-start)
        return result


def _signature_def_to_tensors(signature_def):  # from SeguinBe
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}


def predict(model_dir, image,gpu_id = 0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with tf.Session() as sess:
        start = time.time()
        model = PredictionModel(model_dir,session=sess)
        predictions = model.predict(image)
        transcription = predictions['words']
        score = predictions['score']
        return [transcription[0].decode(), score, time.time() - start]


if __name__ == '__main__':
    model_dir = 'model/'
    image = imread('3_song.jpg', mode='L')[:, :, None]
    result = predict(model_dir, image,0)
    print(tf.__version__)
    print(result)