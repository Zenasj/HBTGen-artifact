from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.python.keras.layers import Conv2D, GlobalAveragePooling2D, Input, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import l2

model = Sequential([
    Input((224, 224, 3)),
    Conv2D(256, (3, 3), kernel_regularizer=l2()),
    GlobalAveragePooling2D(),
    Dense(10, activation='sigmoid'),
])

# Metrics are added
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.metrics)

# Metric are empty
model.compile(optimizer='Adam', metrics=['accuracy'])
print(model.metrics)