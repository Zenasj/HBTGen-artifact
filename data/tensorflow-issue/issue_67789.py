# tf.random.uniform((1, 224, 224, 3), dtype=tf.uint8) ← inferred input shape and dtype from TFLite INT8 preprocessing and resize

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    """
    This MyModel encapsulates the behavior of running inference on a 
    TensorFlow Lite INT8 quantized CenterNet MobileNet model that performs 
    hand keypoint detection, including handling the input preprocessing,
    interpreter inference, and output extraction.
    
    Since the original issue relates to TFLite INT8 runtime errors with tile operations,
    this model simulates the invocation of the TFLite interpreter inside a keras.Model
    interface. Note that this is a conceptual wrapper around the TFLite interpreter, 
    as the original issue is specifically about TFLite runtime.
    """

    def __init__(self, model_path: str):
        super().__init__()
        # Load and initialize the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Assume model expects uint8 inputs with shape [1, 224, 224, 3]
        expected_input_shape = self.input_details[0]['shape']
        expected_input_dtype = self.input_details[0]['dtype']
        # Store for verification
        self.expected_shape = expected_input_shape
        self.expected_dtype = expected_input_dtype

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        Expect inputs as a tf.Tensor of shape [1, height, width, 3] and dtype tf.uint8.
        Resizes inputs to (224, 224) as needed, simulating the resizing done in original code.
        Runs the interpreter and returns all relevant outputs.
        """
        # Resize input to (224, 224) if necessary
        input_resized = tf.image.resize(inputs, (224, 224))
        # Cast and scale the input as expected: TFLite INT8 model expects uint8 input in [0,255].
        if input_resized.dtype != tf.uint8:
            input_processed = tf.cast(input_resized, tf.uint8)
        else:
            input_processed = input_resized

        # Temporarily numpy arrays extraction since TFLite Python API works on numpy
        input_np = input_processed.numpy()

        # Set tensor to interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], input_np)
        self.interpreter.invoke()

        # Extract outputs according to original TFLite model's output tensors by indexes
        boxes = self.interpreter.get_tensor(self.output_details[2]['index'])          # Boxes tensor
        classes = self.interpreter.get_tensor(self.output_details[0]['index'])        # Classes tensor
        scores = self.interpreter.get_tensor(self.output_details[3]['index'])         # Scores tensor
        num_detections = self.interpreter.get_tensor(self.output_details[5]['index']) # Number of detections
        kpts = self.interpreter.get_tensor(self.output_details[1]['index'])           # Keypoints tensor
        kpts_scores = self.interpreter.get_tensor(self.output_details[4]['index'])    # Keypoint scores tensor

        # Convert outputs to TensorFlow tensors for possible further TF pipeline usage
        boxes_tf = tf.convert_to_tensor(boxes)
        classes_tf = tf.convert_to_tensor(classes)
        scores_tf = tf.convert_to_tensor(scores)
        num_detections_tf = tf.convert_to_tensor(num_detections)
        kpts_tf = tf.convert_to_tensor(kpts)
        kpts_scores_tf = tf.convert_to_tensor(kpts_scores)

        # Return tuple mimicking original detect() outputs including keypoints
        # Shape details (from COCO standard detection outputs):
        # boxes: [batch, N, 4], classes: [batch, N], scores: [batch, N], 
        # num_detections: [batch]
        # keypoints: [batch, N, 17, 2], keypoint_scores: [batch, N, 17]
        return boxes_tf, classes_tf, scores_tf, num_detections_tf, kpts_tf, kpts_scores_tf

def my_model_function():
    """
    Returns an instance of MyModel initialized with the TFLite INT8 quantized CenterNet model.
    The model path is assumed to be at '/home/ai_server/Shabbir/DISIGN/model_weights/centernet_int8_03.tflite'
    as per the issue context. Adjust this path if necessary.
    """
    model_path = '/home/ai_server/Shabbir/DISIGN/model_weights/centernet_int8_03.tflite'
    return MyModel(model_path)

def GetInput():
    """
    Returns a random uint8 tensor simulating a batch of input images with shape [1, 224, 224, 3],
    which is the input expected by the model (batch size 1, 224x224 RGB image).
    This input uses uniform random values in [0,255].
    """
    input_tensor = tf.random.uniform(shape=(1, 224, 224, 3), minval=0, maxval=256, dtype=tf.int32)
    input_tensor = tf.cast(input_tensor, tf.uint8)
    return input_tensor

