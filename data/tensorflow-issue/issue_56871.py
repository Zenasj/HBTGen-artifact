import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

def upsambiskip(scale=3, in_channels=3, num_fea=28, m=4, out_channels=3):
    inp = Input(shape=(None, None, 3)) 
    upsampled_inp=UpSampling2D(size=(3,3),data_format=None,interpolation='bilinear')(inp)

    x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

    for i in range(m):
        x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    
    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    out = depth_to_space(x)
    x = Add()([upsampled_inp, out])
    x = Conv2D(3, 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(x)
    
    return Model(inputs=inp, outputs=out)

class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []
    def get_activations_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_anctivations):
        pass
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}

def ps_quantization(self, layer):
    # lambda not quantization
    if 'lambda' in layer.name :
        return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
    return layer


# create fp32 model and load weight
p_model = create_model(args['networks'])
    
lg.info('Start copying weights and annotate Lambda layer...')
annotate_model = tf.keras.models.clone_model(
            p_model,
            clone_function=self.ps_quantization
            )
lg.info('Start annotating other parts of model...')
annotate_model = tfmot.quantization.keras.quantize_annotate_model(annotate_model)
lg.info('Creating quantize-aware model...')
depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, 3))
with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': depth_to_space, 'tf': tf}):
    self.model = tfmot.quantization.keras.quantize_apply(annotate_model)
    
# training...

def qat_quantize(model_path,output_path):

    fakeqmodel=tf.keras.models.load_model(model_path)
    print('fake quantize model validate..')

    # validate fake QAT model, and the performance is OK
    load_validate(fakeqmodel,save_path=None)

    # convert QAT model and store int8 tflite model
    converter = tf.lite.TFLiteConverter.from_keras_model(fakeqmodel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    ### converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.experimental_new_converter=True
    converter.experimental_new_quantizer=True
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quant_tf_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(quant_tf_model)

def evaluate(quantized_model_path, save_path):

    interpreter = tf.lite.Interpreter(model_path=quantized_model_path,num_threads=32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    IS, IZ = input_details[0]['quantization']
    OS, OZ = output_details[0]['quantization']
    print('Input Scale: {}, Zero Point: {}'.format(IS, IZ))
    print('Output Scale: {}, Zero Point: {}'.format(OS, OZ))
    psnr = 0.0
    for i in range(801, 901):
        lr_path = 'data/DIV2K/DIV2K_train_LR_bicubic/X3_pt/0{}x3.pt'.format(i)
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)
        h, w, c = lr.shape
        lr = np.expand_dims(lr, 0).astype(np.float32)
        # ##lr = np.round(lr/IS+IZ).astype(np.uint8)
        lr = lr.astype(np.uint8)

        hr_path = 'data/DIV2K/DIV2K_train_HR_pt/0{}.pt'.format(i)
        with open(hr_path, 'rb') as f:
            hr = pickle.load(f)
        hr = np.expand_dims(hr, 0).astype(np.float32)
        interpreter.resize_tensor_input(input_details[0]['index'], lr.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], lr)
        interpreter.invoke()

        sr = interpreter.get_tensor(output_details[0]['index'])
        #sr = np.clip(np.round((sr.astype(np.float32)-OZ)*OS), 0, 255)
        sr = np.clip(sr, 0, 255)
        b, h, w, c = sr.shape
        # save image
        if save_path is not None:
            save_name = osp.join(save_path, '{:04d}x3.png'.format(i))
            cv2.imwrite(save_name, cv2.cvtColor(sr.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR))

        mse = np.mean((sr[:, 1:h-1, 1:w-1, :].astype(np.float32) - hr[:, 1:h-1, 1:w-1, :].astype(np.float32)) ** 2)
        singlepsnr =  20. * math.log10(255. / math.sqrt(mse))
        print('[{}]/[100]: {}'.format(i, singlepsnr))
        psnr += singlepsnr
    print(psnr / 100)