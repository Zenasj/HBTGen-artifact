converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
# For normal conversion:
#converter.experimental_new_converter = True
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

array([[[ 0.19460055, -0.07871183,  0.348869  , ...,  0.56901807,
          0.40855035, -0.36336952],
        [ 0.111944  , -0.10565656,  0.47841206, ...,  0.7020322 ,
          0.1973463 , -0.56403255],
        [ 0.2462692 , -0.04639575,  0.47352695, ...,  0.6418205 ,
          0.26107264, -0.72952   ],
        ...,
        [ 0.2746247 , -0.06043699,  0.4579026 , ...,  0.7356582 ,
          0.4970783 , -0.6069904 ],
        [ 0.2640199 , -0.05569951,  0.4638841 , ...,  0.734777  ,
          0.48725146, -0.60665435],
        [ 0.25305653, -0.04610543,  0.44723004, ...,  0.7529757 ,
          0.47617406, -0.6056732 ]]], dtype=float32)

from transformers import TFDistilBertModel, DistilBertTokenizer, DistilBertConfig
import tensorflow as tf
from typing import Dict
import numpy as np

class myTFDistilBertModel(TFDistilBertModel):
    # DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
    # tf.constant(np.zeros(shape=(1, 100), dtype=int).tolist())
    # For details refer https://github.com/huggingface/transformers/blob/7cbf0f722d23440f3342aafc27697b50ead5996b/src/transformers/modeling_tf_utils.py#L218
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
    
    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.
        Returns:
            :obj:`Dict[str, tf.Tensor]`: The dummy inputs.
        """
        return {"input_ids": tf.constant(np.zeros(shape=(1, 128), dtype=int).tolist())}

tf_model = myTFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased', return_dict=True)
tf_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased', return_tensors='tf')

inputs = tf_tokenizer("le droit d'accès", padding='max_length', max_length=128, return_tensors='tf')
outputs  = tf_model(inputs)

input_spec = tf.TensorSpec([1, 128], tf.int32)
tf_model._set_inputs(input_spec, training=False)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
# For normal conversion:
#converter.experimental_new_converter = True
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Run the model with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
# Allocate tensors.
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input = tf_tokenizer("le droit d'accès", padding='max_length', max_length=128, return_tensors='tf')
interpreter.set_tensor(0, input['input_ids'])
interpreter.invoke()
interpreter_output = interpreter.get_tensor(output_details[0]["index"])

from transformers import TFDistilBertModel, DistilBertTokenizer, DistilBertConfig
import tensorflow as tf
from typing import Dict
import numpy as np

class myTFDistilBertModel(TFDistilBertModel):
    # DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
    # tf.constant(np.zeros(shape=(1, 100), dtype=int).tolist())
    # For details refer https://github.com/huggingface/transformers/blob/7cbf0f722d23440f3342aafc27697b50ead5996b/src/transformers/modeling_tf_utils.py#L218
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
    
    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.
        Returns:
            :obj:`Dict[str, tf.Tensor]`: The dummy inputs.
        """
        return {"input_ids": tf.constant(np.zeros(shape=(1, 128), dtype=int).tolist())}

tf_model = myTFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased', return_dict=True)
tf_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased', return_tensors='tf')

inputs = tf_tokenizer("le droit d'accès", padding='max_length', max_length=128, return_tensors='tf')
outputs  = tf_model(inputs)

input_spec = tf.TensorSpec([1, 128], tf.int32)
tf_model._set_inputs(input_spec, training=False)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
# For normal conversion:
#converter.experimental_new_converter = True
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Run the model with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
# Allocate tensors.
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input = tf_tokenizer("le droit d'accès", padding='max_length', max_length=128, return_tensors='tf')
interpreter.set_tensor(0, input['input_ids'])
interpreter.invoke()
interpreter_output = interpreter.get_tensor(output_details[0]["index"])

interpreter_output

array([[[ 0.1946005 , -0.07871191,  0.34886912, ...,  0.56901795,
          0.4085501 , -0.36336952],
        [ 0.11194398, -0.1056565 ,  0.47841233, ...,  0.70203197,
          0.19734594, -0.56403244],
        [ 0.24626903, -0.0463957 ,  0.4735275 , ...,  0.64182055,
          0.2610727 , -0.72952026],
        ...,
        [ 0.2746247 , -0.06043694,  0.45790255, ...,  0.73565805,
          0.4970783 , -0.6069902 ],
        [ 0.26401982, -0.05569961,  0.4638846 , ...,  0.7347771 ,
          0.48725143, -0.60665435],
        [ 0.25305668, -0.0461056 ,  0.44722998, ...,  0.7529754 ,
          0.47617412, -0.6056729 ]]], dtype=float32)

outputs

#input_spec = tf.TensorSpec([1, 128], tf.int32)
#tf_model._set_inputs(input_spec, training=False)