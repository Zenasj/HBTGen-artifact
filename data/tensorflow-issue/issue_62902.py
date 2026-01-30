import math
from tensorflow import keras
from tensorflow.keras import layers

# code for convert model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from albumentations import Compose, LongestMaxSize, Normalize, PadIfNeeded
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
input_size = (128, 128)


def preprocess_input(
    input: List[np.ndarray],
    image_size: Tuple[int, int] = (128, 128),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    width, height = image_size
    transforms = Compose(
        [
            LongestMaxSize(max_size=max(image_size)),
            PadIfNeeded(width, height, border_mode=cv2.BORDER_CONSTANT, value=0),
            Normalize(mean=mean, std=std),
        ]
    )
    res = np.array([transforms(image=np.array(item))["image"] for item in input])
    return res


def normalize_func(
    image: np.ndarray,
    mean: Optional[Tuple[float, float, float]] = mean,
    std: Optional[Tuple[float, float, float]] = std,
    max_pixel_value: float = 255.0,
) -> np.ndarray:
    if mean is not None and std is not None:
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        std = tf.convert_to_tensor(std, dtype=tf.float32)
        mean *= max_pixel_value
        std *= max_pixel_value
        denominator = tf.math.reciprocal(std)
        image -= mean
        image *= denominator
    return image


preprocess_input_img_size = lambda x: preprocess_input(x, input_size)  # noqa

    
base_model = EfficientNetV2S(input_shape=(128, 128, 3), include_top=False)
model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(1500, activation="softmax"),
    ]
)

@tf.function(input_signature=[tf.TensorSpec(shape=(1, 128, 128, 3), dtype=tf.uint8)])
def tf_model(image):
    image = tf.cast(image, tf.float32)
    image = normalize_func(image)
    predictions = model(image)
    return predictions


tf_model_func = tf_model.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_model_func])
tflite_model = converter.convert()
tflite_model_file = Path('models/20240201_efficientnetv2s_model.tflite')
tflite_model_file.write_bytes(tflite_model)

# Then I use the model in swift app with CoreML Delegate