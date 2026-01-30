import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras import optimizers, models, layers, callbacks

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name = 'preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = Dropout(0.1)(net)
    net = Dense(1024, activation = 'relu', name = 'hidde')(net)
    net = Dense(256, activation = 'relu', name = 'hidden')(net)
    net = Dense(256, activation = 'relu', name = 'hidden_')(net)
    net = Dense(128, activation = 'relu', name = 'hidden_l')(net)
    net = Dense(64, activation = 'relu', name = 'hidden_la')(net)
    net = Dense(64, activation = 'relu', name = 'hidden_lay')(net)
    net = Dense(16, activation = 'relu', name = 'hidden_laye')(net)
    net = Dense(3, activation='softmax', name='output')(net)
    return tf.keras.Model(text_input, net)
    
def load_callbacks(patience_num, filename):
  return [
    callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = patience_num
    ),
    callbacks.ModelCheckpoint(
        filepath = f'{filename}.h5',
        monitor = 'val_loss',
        save_best_only = True,
        verbose = 1
    )
  ]

loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
metrics = tf.metrics.CategoricalAccuracy()
optimizer = optimizers.RMSprop(learning_rate = 0.001)
epochs = 5
batch_size = round(train_x.shape[0]/10)
batch_size = 1500

model = build_classifier_model()

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

print("Training:")
history = model.fit(x = train_x, y = train_y, epochs = epochs, validation_data = (val_x, val_y), callbacks = load_callbacks(10, 'model'), verbose = 1)
print("Testing:")
model.evaluate(test_x, test_y)

### Relevant log output



gpus = tf.config.experimental.list_physical_devices('GPU')
print()
print(gpus)
print()
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)