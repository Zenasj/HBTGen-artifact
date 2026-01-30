from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
tf.keras.backend.set_floatx('float64')

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class Model1(tf.keras.Model):

  def __init__(self):
    super(Model1,self).__init__(name = 'Model1')
    self.model = models.Sequential()

  def call(self,inputs):
    # self.model.add(layers.Input(shape=(32, 32, 3)))
    self.model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(32,32,3)))
    self.model.add(layers.MaxPooling2D((2, 2)))
    self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    self.model.add(layers.MaxPooling2D((2, 2)))
    self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    return self.model

class Ensemble(tf.keras.Model):
  def __init__(self,model1, model2):
    super(Ensemble,self).__init__(name = 'Ensemble')
    self.model = models.Sequential()
    self.model.add(model1)
    self.model.add(model2)

  def call(self,inputs):
    output = self.model(inputs)
    return output

# class Ensemble(tf.keras.Model):

#   def __init__(self,model1, model2):
#     super(Ensemble,self).__init__(name = 'Ensemble')
#     self.model1 = model1
#     self.model2 = model2
#   def call(self,inputs):
#     output = self.model1(inputs)
#     output = self.model2(output)
#     return output

model1 = Model1()
model2 = Model2()
model = Ensemble(model1,model2)
model.build((32,32,3))
model.summary()

Model: "Ensemble"

# Compile model
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=64,
                      validation_data=(test_images, test_labels))

for i, la in enumerate(model.layers):
  print(la.name)
  for j, laye in enumerate(la.layers):
    print(laye.name)
    for k, lay in enumerate(laye.layers):
      print(lay.name)

model1
sequential
conv2d
max_pooling2d
conv2d_1
max_pooling2d_1
conv2d_2
model2
sequential_1
reshape
flatten
dense
dense_1