from tensorflow.keras import layers

from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Dense, Flatten

input_shape = (110, 110, 3)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
#base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

x = base_model.output
x = Flatten()(x)