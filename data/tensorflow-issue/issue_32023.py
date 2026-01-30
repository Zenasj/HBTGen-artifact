from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model

inputs = Input(shape=(10,))

all_layers = []

x1 = Dense(512)(inputs)
all_layers.append(x1)

# all layers: [x1]
x2 = Dense(256, activation='relu')(x1)
all_layers.append(x2)

# all layers: [x1, x2]
conc = Concatenate()(all_layers)
x3 = Dense(128, activation='relu')(conc)
all_layers.append(x3)

# all layers: [x1, x2, x3]
conc = Concatenate()(all_layers)
prediction = Dense(1)(conc)
model = Model(inputs=inputs, outputs=prediction)

from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model

inputs = Input(shape=(10,))

all_layers = []

x1 = Dense(512)(inputs)
all_layers.append(x1)

# all layers: [x1]
x2 = Dense(256, activation='relu')(x1)
all_layers.append(x2)

# all layers: [x1, x2]
conc = Concatenate()(list(all_layers))
x3 = Dense(128, activation='relu')(conc)
all_layers.append(x3)

# all layers: [x1, x2, x3]
conc = Concatenate()(list(all_layers))
prediction = Dense(1)(conc)
model = Model(inputs=inputs, outputs=prediction)