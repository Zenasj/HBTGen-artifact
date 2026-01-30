from tensorflow import keras

import tensorflow as tf

def main():
    model = tf.keras.Sequential([
        tf.keras.Input(batch_shape=[1, 224, 224, 3]),
        tf.keras.applications.MobileNetV3Small(
            input_shape=[224, 224, 3],
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights="imagenet",
            pooling="max",
            dropout_rate=0.2,
            classifier_activation=None,
            include_preprocessing=True,
        )
    ])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("converted_model.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()

py
import ai_edge_torch
import torch
import torchvision


orig_model = torchvision.models.mobilenet_v3_small()
sample_input = (torch.randn(1, 3, 224, 224),)

edge_model = ai_edge_torch.convert(orig_model.eval(), sample_input)
edge_model.export("mobilenet_v3_small.tflite")

model = ai_edge_torch.to_channel_last_io(model, args=[0])
_args = (
    torch.randn((1, 224, 224, 3), dtype=torch.float32),
)

edge_model = ai_edge_torch.convert(model, _args)
edge_model.export("edge_model.tflite")