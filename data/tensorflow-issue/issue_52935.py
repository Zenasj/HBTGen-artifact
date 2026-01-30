import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

blazeface_tf_model = tf.keras.models.load_model(r"/content/blazeface_tf.h5")
class FaceDetetor(layers.Layer):
    def __init__(self):
        super(FaceDetetor, self).__init__()
        self.classifier = blazeface_tf_model


    def call(self, inputs):
        input_tensor = (inputs / 127.5) - 1
        temp_tensor = self.classifier(input_tensor)
        
        reshape_tensor = tf.reshape(temp_tensor, [-1, temp_tensor.shape[2]])
        final_boxes = tf.slice(reshape_tensor, [0, 0], [-1, 4])
        
        temp_boxex_1 = tf.slice(final_boxes, [0, 0], [-1, 2])  # 中心 y_x
        temp_boxex_2 = tf.slice(final_boxes, [0, 2], [-1, 2])  # w_h
        temp_boxex_2_1 = temp_boxex_2 / 2
        
        ts_sub1 = tf.subtract(temp_boxex_1, temp_boxex_2_1, name=None)
        temp_clip = tf.clip_by_value(ts_sub1, 0, 100000000)
        
        ts_add1 = tf.add(temp_boxex_1, temp_boxex_2_1, name=None)
        
        yx1_columns = tf.unstack(temp_clip, axis=-1)
        xy1_columns = tf.stack([yx1_columns[1], yx1_columns[0]], axis=-1)
        yx2_columns = tf.unstack(ts_add1, axis=-1)
        xy2_columns = tf.stack([yx2_columns[1], yx2_columns[0]], axis=-1)
        box_tlbr = tf.concat([xy1_columns, xy2_columns], 1)
        #### xywh_to_tlbr ####
        raw_scores = tf.slice(reshape_tensor, [0, temp_tensor.shape[2] - 1], [-1, 1])
        scores = tf.reshape(raw_scores, [-1])
        out_boxes = tf.image.non_max_suppression(box_tlbr, scores, max_output_size=5,
                                                 score_threshold=0.5, iou_threshold=0.3)
       
        rows = tf.gather(reshape_tensor, out_boxes, axis=0)
        final_boxes = tf.slice(rows, [0, 0], [-1, temp_tensor.shape[2] - 1])
        if len(final_boxes.shape) == 1:
            final_boxes = tf.expand_dims(final_boxes, axis=0)
        orig_points = final_boxes * 128
        final_scores = tf.gather(raw_scores, out_boxes, axis=0)
        final_result = tf.concat([orig_points, final_scores], 1)
        return final_result
max_output_size = 5
score_threshold = 0.75
iou_threshold = 0.5
inputs_0 = keras.Input(batch_shape=((1, 128, 128, 3)), dtype=tf.float32, name="input_images")
# inputs_1 = keras.Input(shape=(1), dtype=tf.int32, name="max_output_size")
# inputs_2 = keras.Input(shape=(1), dtype=tf.float32, name="score_threshold")
# inputs_3 = keras.Input(shape=(1), dtype=tf.float32, name="iou_threshold")
outputs = FaceDetetor()(inputs_0)

model_folder_path = r"/content/face_detector_inputs3"
model = keras.Model(inputs_0, outputs=outputs)
model.save(model_folder_path, save_format='tf')
print("saved model!")

converter = tf.lite.TFLiteConverter.from_saved_model(model_folder_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
    ]
tflite_model = converter.convert()
output_tflite_path_float = r'/content/face_detector_float16.tflite'
with open(output_tflite_path_float, 'wb') as f:
    f.write(tflite_model)