from tensorflow.keras import layers, Model, Sequential


class ConvBNReLU(layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        layers_list = [layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                     strides=stride, padding='SAME', use_bias=False, name='Conv2d'),
                       layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='BatchNorm'),
                       layers.ReLU(max_value=6.0)]

        self.combine_layer = Sequential(layers_list, name="combine")

    def call(self, inputs, training=False, **kwargs):
        x = self.combine_layer(inputs, training=training)
        return x


def main():
    input_image = layers.Input(shape=(224, 224, 3), dtype='float32')
    # conv1
    x = ConvBNReLU(32, stride=2)(input_image)
    output = ConvBNReLU(64, stride=2)(x)
    model = Model(inputs=input_image, outputs=output)

    for i in model.weights:
        print(i.name)


if __name__ == '__main__':
    main()