import tensorflow as tf

class Net(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=(1,1,3),dtype=tf.uint8)])
    def process(self,image):
        return image

test_exp=Net()

tf.saved_model.save(test_exp,"saved_model")

converter_tlite=tf.lite.TFLiteConverter.from_saved_model("saved_model")

converter_tlite.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

converter_tlite.inference_input_type=tf.uint8
converter_tlite.inference_output_type=tf.uint8

test_tflite=converter_tlite.convert()