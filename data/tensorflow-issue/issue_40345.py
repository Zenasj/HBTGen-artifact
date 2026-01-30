import random
from tensorflow.keras import models

max_features = 10
maxlen = 10
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))

model.add(LSTM(64, go_backwards=True, return_sequences = True))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


from keras.models import load_model
model.save('simple_rev_lstm.h5')

# pip install onnxruntime
#env TF_KERAS=1
import numpy as np
import onnxruntime
from tensorflow.keras.models import load_model as load_model_tf_keras
np.random.seed(0)
input_data = np.random.randint(10, size=(2, 10))
# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
sess = onnxruntime.InferenceSession("simple_rev_lstm.onnx")
result = sess.run(["dense_1"], {'embedding_1_input': input_data.astype(np.float32)})
print("ONNX Runtime")
print(np.asarray(result[0]))

model = load_model_tf_keras('simple_rev_lstm.h5')
result = model.predict(input_data)
print("TF Runtime")
print(result)