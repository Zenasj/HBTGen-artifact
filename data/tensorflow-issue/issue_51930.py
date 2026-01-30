import numpy as np

# relevant import
import tflite_runtime.interpreter as tflite

class DetectionModel(object):
	def __init__(self, path):
		self.interpreter = tflite.Interpreter(model_path=path)
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.input_shape = self.input_details[0]['shape']

	def predict(self, input_img):
		R, C, _ = input_img.shape
		img = cv2.resize(input_img, (self.input_shape[2], self.input_shape[1]))	
		if not self.input_details[0]['dtype'] == np.uint8:
			img = img.astype(np.float32)
			img = (img-128.0)/128.0
		img = np.expand_dims(img, 0)
		self.interpreter.set_tensor(self.input_details[0]['index'], img)			
		self.interpreter.invoke()			
		boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
		classes = self.interpreter.get_tensor(self.output_details[1]['index'])
		scores = self.interpreter.get_tensor(self.output_details[2]['index'])

		return boxes, classes, scores
		
def initialize_classification_interpreter(path):
	interpreter = tflite.Interpreter(model_path=path)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	input_shape = input_details[0]['shape']


	return interpreter, input_details, output_details

def classify(interpreter, input_details, output_details, img):
	
	input_shape = input_details[0]['shape']		

	img = (img/255.0).astype(np.float32)

	img = np.expand_dims(img, 0)
	interpreter.set_tensor(input_details[0]['index'], img)				
	interpreter.invoke()			
	output = interpreter.get_tensor(output_details[0]['index'])
	output = np.squeeze(output)

	return output

detection_model = DetectionModel(detection_path)
interpreter1, input_details1, output_details1 = initialize_classification_interpreter(classification_path_1)
interpreter2, input_details2, output_details2 = initialize_classification_interpreter(classification_path_2)

# warmup: this causes the issue
img = cv2.imread('test_img.jpg')
detection_model.predict(img)
classify(interpreter1, input_details1, output_details1, img)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    # different conditions require different inferences
    if condition_1:
        result = detection_model.predict(img)
    if condition_2:
        result = classify(interpreter1, input_details1, output_details1, img)
    if condition_3:
        result = classify(interpreter2, input_details2, output_details2, img)

    # other operations...