import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

tf.get_logger().setLevel('INFO')

INPUT_SIZE = (512, 5000)
SAVED_MODEL_DIR="/tmp/1234"


if __name__ == "__main__":

    # Create a basic model instance
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4096, activation='relu'),
    ])
    
    # Build model - randomly initialized
    _ = model(tf.random.uniform(INPUT_SIZE))

    # Save Model
    model.save(SAVED_MODEL_DIR)
    
    # =================== INFERENCE ================ #
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=SAVED_MODEL_DIR,
        input_saved_model_tags=[tag_constants.SERVING],
        input_saved_model_signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        precision_mode=trt.TrtPrecisionMode.FP32,
        minimum_segment_size=1
    )

    func = converter.convert()

    converter.summary()

    data = tf.random.uniform(INPUT_SIZE)
    for step in range(1, 20):
        print(f"Step: {step + 1}/50")
        _ = func(data).numpy()