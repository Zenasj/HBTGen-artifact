from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import Model
import traceback


class CustomLayer(Layer):
    def __init__(self, units_base, *args, **kargs):
        super().__init__(self, *args, **kargs)
        self.units_base = units_base
        return

    def build(self, input_shapes):
        self.layer = layers.Dense(self.units_base + input_shapes[-1])
        self.built = True
        return

    def call(self, inputs, *args, **kargs):
        x = self.layer(inputs)
        return x


class CustomModel(Model):
    def __init__(self, *args, **kargs):
        super().__init__(self, *args, **kargs)
        return

    def build(self, input_shapes):
        self.l0 = CustomLayer(input_shapes[-1])
        self.l1 = CustomLayer(input_shapes[-1])
        self.built = True
        return

    def call(self, inputs, *args, **kargs):
        x = self.l0(inputs)
        x = self.l1(x)
        return x


if __name__ == '__main__':
    print('Case1: the model is not built in prior to compute_output_shape')
    try:
        model = CustomModel()
        print('DEBUG: init bult:', model.built)
        outputshape = model.compute_output_shape([100, 50])
        print(outputshape)
    except Exception as e:
        print('Failed')
        traceback.print_exc()

    print('\nCase2: the model is built in prior to compute_output_shape')
    try:
        model = CustomModel()
        model.build([100, 50])
        print('DEBUG: init bult:', model.built)
        outputshape = model.compute_output_shape([100, 50])
        print(outputshape)
    except Exception as e:
        print('Failed')
        traceback.print_exc()