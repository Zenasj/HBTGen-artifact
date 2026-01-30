from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf


############# Create a model using TF and the popular transformers NLP package ###########

class TagModelCreator:

    def __init__(self, language_model):
        self.language_model = language_model

    def create(self, num_classes, max_seq_len, get_token_type_ids=False):

        input_modules = []
        
        input_modules.append(tf.keras.layers.Input(shape=(max_seq_len), dtype='int32', name='input_ids'))
        input_modules.append(tf.keras.layers.Input(shape=(max_seq_len), dtype="int32", name='attention_mask'))

        lang_layer = self.language_model(input_modules)
        linear_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, name='classifier'))(lang_layer[0])
        model = tf.keras.Model(inputs=input_modules, outputs=linear_layer)

        return model

from transformers import TFAutoModel

model_name = "bert-base-uncased"
language_model = TFAutoModel.from_pretrained(model_name)
tagging_model_creator = TagModelCreator(language_model)
arbitrary_class_num = 2
arbitrary_sequence_length = 10
tagging_model = tagging_model_creator.create(arbitrary_class_num, arbitrary_sequence_length)




######### Create some spoof data to see how the model handles the data ####################

def data_generator():
    yield (([0]*arbitrary_sequence_length, [1]*arbitrary_sequence_length))

input_types = ((tf.int32, tf.int32))
input_shape = ((tf.TensorShape([None]), tf.TensorShape([None])))

tf_dataset = tf.data.Dataset.from_generator(data_generator, input_types, input_shape).batch(7)






######### Use the spoof data on the model, to confirm that it does inference on the data without errors########

for example_input in tf_dataset:
    test_output = tagging_model(example_input)
    break

print(test_output)
print("Inference is done correctly BEFORE re-loading the model")






######## Save and reload the model #################

tf.keras.models.save_model(model=tagging_model,
                                       filepath="test_model_save.tf",
                                       save_format="tf",
                                       include_optimizer=True
                                      )

reloaded_model = tf.keras.models.load_model(filepath="test_model_save.tf")






####### Try to repeat the inference as above #########
for example_input in tf_dataset:
    test_output = reloaded_model(example_input)
    break

import sklearn
import sys
from sklearn.pipeline import Pipeline, FeatureUnion
from Transformers import TextTransformer