import random
from tensorflow.keras import layers

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import numpy as np

# Build model
In = Input(shape=(10,))
x = Dense(32)(In)
Out = Dense(2)(x)

# Compile
model = Model(inputs=In, outputs=Out)
model.compile(optimizer='adam', loss='mse')        

# Create dummy input data
fake_data = np.random.uniform(low=0, high=1.0, size=(1, 10, ))

while True:
    # Repeatedly predict:
    model.predict(fake_data) # No memory leak if this line is replaced with "pass"

# custom batched prediction loop to avoid memory leak issues for now in the model.predict call
y_pred_probs = np.empty([len(X_test), VOCAB_SIZE], dtype=np.float32)  # pre-allocate required memory for array for efficiency

BATCH_INDICES = np.arange(start=0, stop=len(X_test), step=BATCH_SIZE)  # row indices of batches
BATCH_INDICES = np.append(BATCH_INDICES, len(X_test))  # add final batch_end row

for index in np.arange(len(BATCH_INDICES) - 1):
    batch_start = BATCH_INDICES[index]  # first row of the batch
    batch_end = BATCH_INDICES[index + 1]  # last row of the batch
    y_pred_probs[batch_start:batch_end] = model.predict_on_batch(X_test[batch_start:batch_end])