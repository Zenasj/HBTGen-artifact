import time
import tensorflow as tf

_INPUT_KEYS_TO_TENSOR       = "input_keys_to_tensor"
_OUTPUT_KEYS_KEYS_TO_TENSOR = "output_keys_to_tensor"

class TF1Model:

  def __init__(self, model_path):
    self._sess = tf.compat.v1.Session()

    if hasattr(tf.compat.v1.saved_model, "load"):
      graph_meta_def = tf.compat.v1.saved_model.load(
        self._sess, [tf.saved_model.SERVING], model_path
      )
    else:
      graph_meta_def = tf.compat.v1.saved_model.loader.load(
        self._sess, [tf.saved_model.SERVING], model_path
      )

    signature = graph_meta_def.signature_def

    self._signature_tensor_mapping = {}
    for signature_name in signature.keys():
      indiv_sig_data = self._signature_tensor_mapping[signature_name] = {
        _INPUT_KEYS_TO_TENSOR: {},
        _OUTPUT_KEYS_KEYS_TO_TENSOR: {}
      }

      inputs = signature[signature_name].inputs
      for k in inputs.keys():
        tensor = self._sess.graph.get_tensor_by_name(inputs[k].name)
        indiv_sig_data[_INPUT_KEYS_TO_TENSOR][k] = tensor

      outputs = signature[signature_name].outputs
      for k in outputs.keys():
        tensor = self._sess.graph.get_tensor_by_name(outputs[k].name)
        indiv_sig_data[_OUTPUT_KEYS_KEYS_TO_TENSOR][k] = tensor

  def predict(self, payload):
    start = time.time()
    payload_sig = payload["signature_name"]
    res = self._sess.run(
      self._signature_tensor_mapping[payload_sig][_OUTPUT_KEYS_KEYS_TO_TENSOR],
      {
        self._signature_tensor_mapping[payload_sig][_INPUT_KEYS_TO_TENSOR]["inputs"]: payload["inputs"]["inputs"]
      }
    )
    print("prediction took: {}s".format(time.time() - start))
    return res

if __name__ == "__main__":

  model = TF1Model("/tf1_models/object_detection/1")

  payload = {
    "signature_name":"serving_default",
    "inputs":{
      "inputs":[
        [[[0, 0, 0], [0, 0, 0]]]
      ]
    }
  }

  for _ in range(3):
    model.predict(payload)