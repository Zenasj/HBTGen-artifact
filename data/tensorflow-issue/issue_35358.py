import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

with tf.io.TFRecordWriter(path='./MNIST.tfrecords') as tf_writer:        
    for image, label in zip(x_train, y_train):
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(image).numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))            
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_writer.write(example.SerializeToString())

datasets_tfrecord = tf.data.TFRecordDataset('MNIST.tfrecords')

feature_type = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def parser_tfrecord(tfrecord_example):
    _feature = tf.io.parse_single_example(tfrecord_example, feature_type)
    _feature['image'] = tf.io.parse_tensor(_feature['image'], tf.float64)
    return _feature['image'], _feature['label']

datasets = datasets_tfrecord.map(parser_tfrecord)

for image, label in datasets.take(1):
    plt.imshow(image.numpy()[:, :, 0])
    plt.title(label.numpy())
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='tanh'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='tanh'),
    tf.keras.layers.Dense(84, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.fit(datasets,
          epochs=5, 
          steps_per_epoch=int(len(x_train)/128))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(y, predictions)

datasets = datasets.batch(64)

for epoch in range(1):  
    for x, y in datasets:
        train_step(x, y)

for x, y in datasets:
    print(x.shape)
    break

# (64, 28, 28, 1)

datasets = datasets.batch(64)

for x, y in datasets:
    print(x.shape)
    break

# (64, 64, 28, 28, 1)

def parser_tfrecord(tfrecord_example):
    _feature = tf.io.parse_single_example(tfrecord_example, feature_type)
    _feature['image'] = tf.io.parse_tensor(_feature['image'], tf.float64)
   # add this line
    _feature['image'].set_shape((28,28,1))
    return _feature['image'], _feature['label']