import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

opt = tf.keras.optimizers.Adam()
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
model.compile(loss=loss, optimizer=opt)
model.fit(...)

tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
model = Sequential([
            Input((28,28,1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(64, (3, 3), activation='relu'),
            GlobalAveragePooling2D(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Dense(10, activation='softmax', dtype=tf.float32)