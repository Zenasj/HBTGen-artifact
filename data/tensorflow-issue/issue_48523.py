import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

demo_class01: [NaN, NaN, NaN, NaN]
demo_class02: [NaN, NaN, NaN, NaN]
demo_class03: [NaN, NaN, NaN, NaN]
demo_class04: [NaN, NaN, NaN, NaN]

def create_demo():

    demo_ins = tf.keras.layers.Input(shape=(300, 300, 3), name="demo_ins", batch_size=1)
    
    model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(300, 300, 3), 
                                              pooling=None, input_tensor=demo_ins)
    out_relu = model.get_layer(name="out_relu").output
    
    demo_conv01 = tf.keras.layers.Conv2D(4, (3, 3), padding="same", strides=(1, 1), 
                                         kernel_initializer="he_uniform", name="demo_conv01")(out_relu)
    demo_bn01   = tf.keras.layers.BatchNormalization(name="demo_bn01")(demo_conv01)
    demo_softmax01 = tf.keras.layers.Softmax(axis=2, name="a_demo_softmax01")(demo_bn01)
    
    
    demo_slice01 = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0], name="demo_slice01")(demo_softmax01)
    demo_slice02 = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1], name="demo_slice02")(demo_softmax01)
    demo_slice03 = tf.keras.layers.Lambda(lambda x: x[:, :, :, 2], name="demo_slice03")(demo_softmax01)
    demo_slice04 = tf.keras.layers.Lambda(lambda x: x[:, :, :, 3], name="demo_slice04")(demo_softmax01)
    
    demo_slice01_reshape = tf.keras.layers.Reshape([10, 10, 1], name="demo_slice01_reshape")(demo_slice01)
    demo_slice02_reshape = tf.keras.layers.Reshape([10, 10, 1], name="demo_slice02_reshape")(demo_slice02)
    demo_slice03_reshape = tf.keras.layers.Reshape([10, 10, 1], name="demo_slice03_reshape")(demo_slice03)
    demo_slice04_reshape = tf.keras.layers.Reshape([10, 10, 1], name="demo_slice04_reshape")(demo_slice04)
    
    demo_mul01 = tf.keras.layers.Multiply(name="w_demo_mul01")([out_relu, demo_slice01_reshape])
    demo_mul02 = tf.keras.layers.Multiply(name="x_demo_mul02")([out_relu, demo_slice02_reshape])
    demo_mul03 = tf.keras.layers.Multiply(name="y_demo_mul03")([out_relu, demo_slice03_reshape])
    demo_mul04 = tf.keras.layers.Multiply(name="z_demo_mul04")([out_relu, demo_slice04_reshape])
    
    demo_block_conv01 = tf.keras.layers.Conv2D(3, (3, 3), padding="same", strides=(1, 1), 
                                                   kernel_initializer="he_uniform", name="demo_block_conv01")
    demo_block_bn01   = tf.keras.layers.BatchNormalization(name="demo_block_bn01")
    demo_block_relu01 = tf.keras.layers.ReLU(name="demo_block_relu01")
    demo_block_flatten = tf.keras.layers.Flatten(name="demo_block_flatten")
    demo_block_softmax01 = tf.keras.layers.Dense(4, activation="softmax", name="demo_block_softmax01")
    
    def block(inputs):
        
        conv01  = demo_block_conv01(inputs)
        bn01    = demo_block_bn01(conv01)
        relu01  = demo_block_relu01(bn01)
        flatten = demo_block_flatten(relu01)
        softmax01 = demo_block_softmax01(flatten)
        
        return softmax01
        
    demo_class01 = tf.keras.layers.Layer(name="b_demo_class01")(block(demo_mul01))
    demo_class02 = tf.keras.layers.Layer(name="c_demo_class02")(block(demo_mul02))
    demo_class03 = tf.keras.layers.Layer(name="d_demo_class03")(block(demo_mul03))
    demo_class04 = tf.keras.layers.Layer(name="e_demo_class04")(block(demo_mul04))
    
    model = tf.keras.models.Model(
        inputs=[model.input], 
        outputs=[demo_softmax01, 
                 demo_class01, demo_class02, demo_class03, demo_class04, 
                 demo_mul01, demo_mul02, demo_mul03, demo_mul04]
    )
    
    return model

model = create_demo()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("demo.tflite", 'wb') as f:
    f.write(tflite_model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

gpuOptions.setQuantizedModelsAllowed(...)
gpuOptions.setInferencePreference(...)

model = create_demo()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("demo.tflite", 'wb') as f:
    f.write(tflite_model)

one = tf.constant([[[[1.0]]]])
out_relu  = tf.keras.layers.Multiply()([out_relu, one])

demo_mul01 = tf.keras.layers.Multiply(name="w_demo_mul01")([out_relu, demo_slice01_reshape])
demo_mul02 = tf.keras.layers.Multiply(name="x_demo_mul02")([out_relu, demo_slice02_reshape])
demo_mul03 = tf.keras.layers.Multiply(name="y_demo_mul03")([out_relu, demo_slice03_reshape])
demo_mul04 = tf.keras.layers.Multiply(name="z_demo_mul04")([out_relu, demo_slice04_reshape])