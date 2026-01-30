from tensorflow.keras import layers

self.model = keras.Sequential([
    keras.layers.Dense(1, input_dim=self.degree),
    keras.layers.Dense(1)
    ])
self.model.compile(optimizer=optimizer, loss=loss)
self.model.summary()

class your_class_name:
	def __init__(self):
		my_code_goes_here