import numpy as np
import tensorflow as tf

class TFInferModel(tf.Module):
    def __init__(self):
        super(TFInferModel, self).__init__()
        self.feat_gen = tf.saved_model.load("feat_gen.pb")
        self.model = tf.saved_model.load("model.pb")

        self.feat_gen.trainable = False
        self.model.trainable = False

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 126], dtype=tf.float32, name="inputs")])
    def call(self, inputs):
        output_tensors = {}

        # Add batch dimension.
        inputs = inputs[None]

        # Process using ported PyTorch model.
        features = self.feat_gen(inputs=inputs)["outputs"]
        outputs = self.model(inputs=features)["outputs"]

        # Remove batch dimension.
        outputs = outputs[0]

        output_tensors["outputs"] = outputs
        return output_tensors

tf_model = TFInferModel()
tf.saved_model.save(tf_model, "tf_model", signatures={"serving_default": tf_model.call})

model = tf.saved_model.load("tf_model")
inputs = tf.zeros((100, 126), dtype=tf.float32)
output = model.call(inputs=inputs)

converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")

tf_lite_model = converter.convert()
output_path = "model.tflite"
with open(output_path, "wb") as f:
    f.write(tf_lite_model)

interpreter = tflite.Interpreter(model_path="model.tflite")
prediction_fn = interpreter.get_signature_runner("serving_default")
inputs = np.zeros((100, 126), dtype=np.float32)
output = prediction_fn(inputs=inputs)