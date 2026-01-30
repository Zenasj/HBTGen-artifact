import tensorflow as tf
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, TFWhisperForConditionalGeneration
import soundfile as sf
import numpy as np
class GenerateModel2(tf.Module):
  def __init__(self, model):
    super(GenerateModel2, self).__init__()
    self.model = model
  @tf.function(
    input_signature=[
      tf.TensorSpec((1, 80, 3000), tf.float32, name="input_ids"),
    ]
  )
  def serving(self, input_features):
    outputs = self.model.generate(input_features)
    return outputs

# Initialize the feature extractor, tokenizer, and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en", predict_timestamps=True)
processor = WhisperProcessor(feature_extractor, tokenizer)

# Load the Whisper model
model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
generate_model = GenerateModel2(model=model)

def representative_data_gen():
  audio_input, sample_rate = sf.read("story-1-second-2.wav")
  inputs = processor(audio_input, return_tensors="tf")
  yield {"input_ids": inputs.input_features}

converter = tf.lite.TFLiteConverter.from_concrete_functions([generate_model.serving.get_concrete_function()], generate_model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
  tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_data_gen
tflite_model_bin = converter.convert()    # <---- crash here

sample_rate= 16000