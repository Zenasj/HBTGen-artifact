import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

model.save()

input_layer = Input(shape = (512,), dtype='int64') 
bert = TFBertModel.from_pretrained('bert-base-chinese')(input_layer)
bert = bert[0]   
dropout = Dropout(0.1)(bert)
flat = Flatten()(dropout)
classifier = Dense(units=5, activation="softmax")(flat)               
model = Model(inputs=input_layer, outputs=classifier)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.save('model/my_model.h5')

tf.keras.models.save_model()

tf.keras.models.save_model(
    model,
    "model/model_bert_eland_softmax_2",
    overwrite=True,
    include_optimizer=True,
)

input_layer = Input(shape = (512,), dtype='int64')  
load_model = tf.keras.models.load_model('model/model_bert_eland_softmax_2')(input_layer)
new_model = Model(inputs=input_layer, outputs=load_model)

# Show the model architecture
new_model.summary()

tf.keras.models.save_model()

model.save()

model.save("my_model",save_format='tf')
loaded_model = tf.keras.models.load_model("my_model")
loaded_model.summary()

model.save("my_model",save_format='tf')
loaded_model = tf.keras.models.load_model("my_model")
loaded_model.summary()