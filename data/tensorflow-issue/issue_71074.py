import numpy as np
import tensorflow as tf

class tflite_instance:
    def __init__(self, model_name, qat=True):
        self.interpreter = tf.lite.Interpreter(model_path=os.path.join(TFLITE_MODELS_DIR, model_name + '.tflite'))
        self.interpreter.allocate_tensors()
        self.tflite_input_details = self.interpreter.get_input_details()
        self.tflite_output_details = self.interpreter.get_output_details()

    def inference(self, x, num_out=1):
        input_details = self.tflite_input_details[0]
        tensor_index = input_details['index']
        input_tensor = self.interpreter.tensor(tensor_index)()
        scale, zero_point = input_details['quantization']        
        quantized_input = np.uint8(x / scale + zero_point)                    
        input_tensor = quantized_input

    
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.tflite_output_details[0]['index'])
                   
        scale, zero_point = self.tflite_output_details[0]['quantization']    
        output = scale * (output.astype(np.float32) - zero_point)                    
        
        return output

class tflite_instance:
    def __init__(self, model_name, qat=True):
        self.interpreter = tf.lite.Interpreter(model_path=os.path.join(TFLITE_MODELS_DIR, model_name + '.tflite'))
        self.interpreter.allocate_tensors()
        self.tflite_input_details = self.interpreter.get_input_details()
        self.tflite_output_details = self.interpreter.get_output_details()
    

    def set_input_tensor(self, x):
        input_details = self.interpreter.get_input_details()[0]
        tensor_index = input_details['index']
        input_tensor = self.interpreter.tensor(tensor_index)()
        # Inputs for the TFLite model must be uint8, so we quantize our input data.
        scale, zero_point = input_details['quantization']
        quantized_input = np.uint8(x / scale + zero_point)
        input_tensor[:, :, :, :] = quantized_input

    def inference(self, x, num_out=1):

        self.set_input_tensor(x)
    
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.tflite_output_details[0]['index'])
                   
        scale, zero_point = self.tflite_output_details[0]['quantization']    
        output = scale * (output.astype(np.float32) - zero_point)                    
        
        return output