def failed_on_android_gpu_for_dynamic_size():
  ipt = layers.Input((256, 64, 3), batch_size=1)
  x = ipt
  x = layers.Reshape((16384, 3))(x)
  x = layers.Softmax(axis=1)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(1)(x)
  model = keras.Model(inputs=[ipt], outputs=[x])
  convert_to_lite(model, "failed_on_android_gpu.tflite")