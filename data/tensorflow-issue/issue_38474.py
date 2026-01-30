import numpy as np
import tensorflow as tf
from tensorflow import keras

def preprocess_image_cv2(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28)).astype("float32")
    img = img / 255
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return img

# Define the model for predcition purpose
class ExportModel(tf.keras.Model):
    def __init__(self, preproc_func, model):
        super().__init__(self)
        self.preproc_func = preproc_func
        self.model = model

    @tf.function
    def my_serve(self, image_path):
        print("Inside")
        preprocessed_image = self.preproc_func(image_path) # Preprocessing
        probabilities = self.model(preprocessed_image, training=False) # Model prediction
        class_id = tf.argmax(probabilities[0], axis=-1) # Postprocessing
        return {"class_index": class_id}

# Now initialize a dummy model and fill its parameters with that of
# the model we trained
restored_model = get_training_model()
restored_model.set_weights(apparel_model.get_weights())

# Now use this model, preprocessing function, and the same image
# for checking if everything is working
serving_model = ExportModel(preprocess_image_cv2, restored_model)
class_index = serving_model.my_serve("sample_image.png")
CLASSES[class_index["class_index"].numpy()] # prints Dress

# Make sure we are *not* letting the model to train
tf.keras.backend.set_learning_phase(0)

# Serialize model
export_path = "model_preprocessing_func"
tf.saved_model.save(serving_model, export_path, signatures={"serving_default": serving_model.my_serve})