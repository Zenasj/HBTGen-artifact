# tf.random.uniform((1, 224, 224, 3), dtype=tf.uint8) ← inferred input shape and dtype from TFLite model input and usage

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    """
    A wrapper Keras Model representing the TFLite INT8 quantized CenterNet MobileNet model
    for hand keypoint detection.
    
    Since TFLite model is used via the Interpreter API, this class emulates the inference step.
    The forward method runs the TFLite interpreter on the input tensor and returns the outputs.
    
    This design allows usage in TF2 environment with tf.function and XLA compilation by
    embedding the interpreter call within the model call.
    
    Input:
        input_tensor: tf.Tensor with shape [1, 224, 224, 3], dtype=tf.uint8
    
    Output:
        Tuple of detection outputs:
          boxes: numpy array shape [N, 4]
          classes: numpy array shape [N]
          scores: numpy array shape [N]
          num_detections: scalar count
          kpts: numpy array shape [N, 17, 2]
          kpts_scores: numpy array shape [N, 17]
    """
    def __init__(self, tflite_model_path='centernet_int8_03.tflite'):
        super().__init__()
        # Load and initialize TFLite interpreter once at object creation
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.uint8)])
    def call(self, input_tensor):
        # The TFLite interpreter expects uint8 input scaled 0-255.
        # Input here assumed to be uint8 already; enforce this.
        # Note: Because TFLite interpreter is a Python object with non-TF ops,
        # we cannot run interpreter.invoke() inside @tf.function directly.
        # To support tf.function + jit_compile=True XLA, we must use tf.py_function.

        def tflite_inference(input_np):
            # input_np is a numpy array with uint8 input
            self.interpreter.set_tensor(self.input_details[0]['index'], input_np)
            self.interpreter.invoke()

            # Extract outputs from the TFLite model interpreter
            classes = self.interpreter.get_tensor(self.output_details[0]['index'])
            kpts = self.interpreter.get_tensor(self.output_details[1]['index'])
            boxes = self.interpreter.get_tensor(self.output_details[2]['index'])
            scores = self.interpreter.get_tensor(self.output_details[3]['index'])
            kpts_scores = self.interpreter.get_tensor(self.output_details[4]['index'])
            num_detections = self.interpreter.get_tensor(self.output_details[5]['index'])
            return boxes, classes, scores, num_detections, kpts, kpts_scores

        # Use tf.py_function to call the numpy-backed TFLite interpreter
        boxes, classes, scores, num_detections, kpts, kpts_scores = tf.py_function(
            func=tflite_inference,
            inp=[input_tensor],
            Tout=[
                tf.float32,      # boxes
                tf.float32,      # classes
                tf.float32,      # scores
                tf.float32,      # num_detections (usually a scalar float)
                tf.float32,      # kpts
                tf.float32,      # kpts_scores
            ])

        # Set static shape info on outputs (optional but helpful)
        boxes.set_shape([None, 4])
        classes.set_shape([None])
        scores.set_shape([None])
        num_detections.set_shape([])
        kpts.set_shape([None, 17, 2])
        kpts_scores.set_shape([None, 17])

        return boxes, classes, scores, num_detections, kpts, kpts_scores

def my_model_function():
    """
    Returns an instance of MyModel initialized with the TFLite INT8 quantized model.
    """
    # Path to the TFLite INT8 quantized model, adjust if needed
    model_path = 'centernet_int8_03.tflite'
    return MyModel(tflite_model_path=model_path)

def GetInput():
    """
    Returns a dummy input tensor compatible with MyModel call.
    The input tensor shape is [1, 224, 224, 3] with dtype uint8,
    simulating a single input image resized and quantized (0-255).
    """
    # Generate a random uint8 tensor with shape matching the TFLite model input
    return tf.random.uniform(shape=(1, 224, 224, 3), minval=0, maxval=256, dtype=tf.uint8)

