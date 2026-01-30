from tensorflow.keras import layers
from tensorflow.keras import models

#! /usr/bin/env python3

import tensorflow
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model

from io import BytesIO
import h5py

inputLayer = Input(batch_shape=(None, 5, 10))
final = Dense(10)(inputLayer)

model = Model(inputs=[ inputLayer ], outputs=[ final ])
opt = tensorflow.optimizers.get({
    'class_name': 'Adam',
    'config': {}
});
model.compile(opt, loss='mean_squared_error', metrics=['mse'])

# Serialization here works fine
with h5py.File('does not matter', driver='core', backing_store=False) as h5file:
    model.save(h5file)
    h5file.flush()
    serialized = h5file.id.get_file_image().hex()

# Deserialization here throws error for tf >= 2.0.0-beta0
restored = load_model(BytesIO(bytes.fromhex(serialized)))