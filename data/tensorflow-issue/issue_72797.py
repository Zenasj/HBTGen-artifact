import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# Example input data
X_train_dict = {
    'green_fin_const': np.random.rand(558, 3),
    'green_fin_inst': np.random.rand(558, 4),
    'gov_sup': np.random.rand(558, 5),
    'com_act': np.random.rand(558, 5),
    'eco_city': np.random.rand(558, 4),
    'type': np.random.rand(558, 1)
}
Y_train = np.random.rand(558, 2)

X_test_dict = {
    'green_fin_const': np.random.rand(140, 3),
    'green_fin_inst': np.random.rand(140, 4),
    'gov_sup': np.random.rand(140, 5),
    'com_act': np.random.rand(140, 5),
    'eco_city': np.random.rand(140, 4),
    'type': np.random.rand(140, 1)
}
Y_test = np.random.rand(140, 2)

# Define input layers
inputs_1 = Input(shape=(3,), name='green_fin_const')
inputs_2 = Input(shape=(4,), name='green_fin_inst')
inputs_3 = Input(shape=(5,), name='gov_sup')
inputs_4 = Input(shape=(5,), name='com_act')
inputs_5 = Input(shape=(4,), name='eco_city')
inputs_6 = Input(shape=(1,), name='type')

# Define dense layers
score_1 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())(inputs_1)
score_2 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())(inputs_2)
score_3 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())(inputs_3)
score_4 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())(inputs_4)
score_5 = Dense(1, activation='sigmoid', kernel_constraint=non_neg())(inputs_5)

# Concatenate scores and type input
concatenated_scores = concatenate([score_1, score_2, score_3, score_4, score_5, inputs_6])

# Define output layer
outputs = Dense(2, activation='softmax', kernel_constraint=non_neg())(concatenated_scores)

# Create the model
model = Model(inputs=[inputs_1, inputs_2, inputs_3, inputs_4, inputs_5, inputs_6], outputs=outputs)

# Plot model architecture
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# Compile model
model.compile(optimizer='nadam', loss='mse', metrics=['KLDivergence'])

# Print input shapes to verify
for key, value in X_train_dict.items():
    print(f'{key}: {value.shape}')

# Fit the model
model.fit(
    x=X_train_dict,
    y=Y_train,
    validation_data=(X_test_dict, Y_test),
    epochs=100,
    batch_size=32,
    verbose=0
)