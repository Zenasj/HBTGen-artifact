from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
print(tf.__version__)

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

samples = np.array([
    [u'Россия', 0],
    [u'Вчера смотрел в кино - потрясающий фильм! Актёры высшие, невероятные декорации, безудержный драйв на протяжении всего фильма. Давно не испытывал такого восторга от просмотра! 10/10', 1],
    [u'Норм фильм,в своём стиле не понимаю что другие ожидали))одно смутило когда сцена в клубе все танчили пока бойня была типо ниче не замечая а как картежника завалили все с истериками побежали,типа хуясе тут все в настаящую))))да и пёсель зачетный))', 1],
    [u'Да пипец блин, меня хватило на 10 минут. Это днище', 0],
    [u'Бредовый фильм не советую', 0],
])

test = np.array([
    [u'Фильм говно', 0],
    [u'Классный фильм', 1],
    [u'Не советую к просмотру', 0],
    [u'Тупой фильм', 0],
])

train_text = []
train_label = []

test_text = []
test_label = []

for sample in samples:
    train_text.append(sample[0])
    train_label.append(float(sample[1]))

for tst in test:
    test_text.append(tst[0])
    test_label.append(float(tst[1]))

dataset = {'train': 0, 'test': 0}

dataset['train'] = tf.data.Dataset.from_tensor_slices((train_text, train_label))
dataset['test'] = tf.data.Dataset.from_tensor_slices((test_text, test_label))

train_dataset, test_dataset = dataset['train'], dataset['test']

for text, lable in train_dataset.take(2):
    print(text)

BUFFER_SIZE = 10000
BATCH_SIZE = 128

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 20000
encoder = tf.keras.layers.TextVectorization(
    standardize='lower',
    max_tokens=VOCAB_SIZE,
    encoding='utf-8')
encoder.adapt(train_dataset.map(lambda text, label: text))


model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=250,
                    validation_data=test_dataset)
                

sample_text = 'меня хватило на 10 минут'
predictions = model.predict(np.array([sample_text]))
print(predictions)