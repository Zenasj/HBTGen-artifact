from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2,activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['AUC'] )

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# Set early stopping
es = EarlyStopping(monitor = 'val_auc', min_delta = 1e-4, patience = 30, mode = 'max', 
                        baseline = None, restore_best_weights = True, verbose = 1)

rlr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.1, patience = 25, verbose = 1, 
                            min_delta = 1e-4, mode = 'max', min_lr = 0.00001)



model.fit(x=X_train, 
          y=y_train.values, 
          epochs=200,
          validation_data=(X_test, y_test.values), 
          verbose=1,
          callbacks=[es,rlr]
          )