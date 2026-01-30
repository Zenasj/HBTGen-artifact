from tensorflow.keras import Model, layers

class BottleneckLayer(layers.Layer):
    def __init__(self, growthRate):
        super().__init__()
        self.conv1 = layers.Conv2D(4 * growthRate, kernel_size=1, strides=1, padding="same")
        self.conv2 = layers.Conv2D(growthRate, kernel_size=3, strides=1, padding="same")
        self.batchNorm = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.relu = layers.Activation("relu")

    def call(self, x):
        y = self.batchNorm(self.relu(self.conv1(x)))
        y = self.batchNorm(self.relu(self.conv2(y)))
        y = layers.concatenate([x, y])
        return y

class DenseNet(Model):
    def __init__(self):
        super().__init__()
        self.relu = layers.Activation("relu")
        growthRate = 12
        self.conv1 = layers.Conv2D(2 * growthRate, kernel_size=7, strides=2, padding="same")
        self.maxpool = layers.MaxPooling2D((2, 2), strides=2) 

        self.bottleneck = BottleneckLayer(growthRate)

    def call(self, x):
        y = self.maxpool(self.relu(self.conv1(x)))
        print(y.shape)

        ## putting BottleneckLayer directly in call works
        for _ in range(6):
            y = BottleneckLayer(12)(y)
            print(y.shape)

        ## this approach does not work
        for _ in range(6):
            y = self.bottleneck(y)
            print(y.shape)

        return y