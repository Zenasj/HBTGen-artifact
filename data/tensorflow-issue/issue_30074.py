class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    #self.conv1 = Conv2D(32, 3, activation='relu')
    self.conv1 = Conv2D(32, kernel_size=3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


model = MyModel()
model1 = model
# above is creating model1 with same number of layers of model.
# but in the following, model1 is coming up with zero layer and model is having 3 layers.
model1 = MyModel()