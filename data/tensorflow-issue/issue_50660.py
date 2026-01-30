import tensorflow as tf

saved_model_obj = tf.saved_model.load(export_dir=model_path)

# Load the specific concrete function from the SavedModel.
concrete_func = saved_model_obj.signatures['serving_default']
# Set the shape of the input in the concrete function.
concrete_func.inputs[0].set_shape((None,))

# Convert the model to a TFLite model.
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model.tflite','wb') as f:
    f.write(tflite_model)

model = tf.saved_model.load(model_path)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=model_dir,signature_keys=['serving_default','predict','classification'])
tflite_model = converter.convert()

with open('model.tflite','wb') as f:
    f.write(tflite_model)

model=XYZModel()
model.load_weights(path)
model.compute_output_shape(input_shape=(None, 240, 320, 4))  # or try model.build(input_shape)

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch. 
    self.__call__.get_concrete_function(
        audio_input=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       audio_input=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


  @tf.function
  def __call__(self, audio_input):
    # If they pass a string, load the file and decode it. 
    if audio_input.dtype == tf.string:
      audio_input = tf.io.read_file(audio_input)
      audio_input, _ = tf.audio.decode_wav(audio_input, desired_channels=1, desired_samples=16000,)
      audio_input = tf.squeeze(audio_input, axis=-1)
      audio_input = audio_input[tf.newaxis, :]
    
    audio_input = get_spectrogram(audio_input)  
    result = self.model(audio_input, training=False)
    
    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}

x = tf.io.read_file(tf.constant(str(data_dir/'no/01bb6a2a_nohash_0.wav')))
x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
x = tf.squeeze(x, axis=-1)
x = x[tf.newaxis, :]
audio_input=x
export(audio_input)

saved_model_dir="/path/to/mydir/"

no_signature_path = os.path.join(saved_model_dir, 'module_with_no_signature')
module_with_signature_path = os.path.join(saved_model_dir, 'module_with_signature')
module_multiple_signatures_path = os.path.join(saved_model_dir, 'module_with_multiple_signatures')

tf.saved_model.save(export, no_signature_path)

call = export.__call__.get_concrete_function(tf.constant(str(data_dir/'no/01bb6a2a_nohash_0.wav')))
tf.saved_model.save(export, module_with_signature_path, signatures=call)

signatures = {"wav_file": call,
              "wav_pcm": export.__call__.get_concrete_function(tf.TensorSpec([1,16000], tf.float32))}
tf.saved_model.save(export, module_multiple_signatures_path, signatures=signatures)

imported = tf.saved_model.load(no_signature_path)
print (imported.signatures)
print (list(imported.signatures.keys()))

imported = tf.saved_model.load(module_with_signature_path)
print (imported.signatures)
print (list(imported.signatures.keys()))

imported = tf.saved_model.load(module_multiple_signatures_path)
print (imported.signatures)
print (list(imported.signatures.keys()))

saved_model_obj = tf.saved_model.load(module_multiple_signatures_path)
concrete_func = saved_model_obj.signatures['wav_pcm']
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

tflite_model = converter.convert()
with open('model-sig.tflite', 'wb') as f:
  f.write(tflite_model)