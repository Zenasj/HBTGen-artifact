from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow import saved_model
from pathlib import Path
from numpy.random import rand

inputs = Input(shape=(200,))
x = Dense(64, activation="relu")(inputs)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs, name="DummyTF")
model.compile()

data = rand(1, 200).tolist()
model.predict(data)

path = str(Path(__file__).parent.resolve() / "ai_model" / "models" / "dummy_tf")

model.save(path)

model_loaded = saved_model.load(path)
model_loaded.predict(data)