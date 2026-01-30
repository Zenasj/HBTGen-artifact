from tensorflow.keras import layers
from tensorflow.keras import models

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# rest of your code

# Normalize the data
scaler = StandardScaler()
train_tables_windows = scaler.fit_transform(train_tables_windows).astype('float32')
test_tables_windows = scaler.transform(test_tables_windows).astype('float32')

# Define the model
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True, input_shape=(128, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.compile(loss='mse', optimizer='adam')
    return model

# Define the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
tensorboard = TensorBoard(log_dir=os.path.join('logs'))
csv_logger = CSVLogger('training.log')

callbacks = [early_stopping, model_checkpoint, tensorboard, csv_logger]

# Fit the model
tuner.search(train_tables_windows, train_labels_windows, epochs=50, validation_split=0.2, callbacks=callbacks)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)

# Fit the model
model.fit(train_tables_windows, train_labels_windows, epochs=50, batch_size=32, callbacks=callbacks)

# Evaluate the model
mse = model.evaluate(test_tables_windows, test_labels_windows)
predictions = model.predict(test_tables_windows)

print('Mean Squared Error:', mse)

# Calculate R2 score
r2 = r2_score(test_labels_windows, predictions)
print('R2:', r2)

# Plot the predicted angles against the true angles
for i in range(10):
    plt.figure(figsize=(16,3))
    plt.plot(predictions[1:200,i], label='Predicted')
    plt.plot(np.array(test_labels_windows.iloc[1:200, i]), label='True')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.title('Predicted vs True Angles of Joint {}'.format(i))
    plt.legend()
    plt.show()