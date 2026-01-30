from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

sinput = Input(shape=(1,),  name='state_input', dtype='int32')
sinput_em = Embedding(output_dim=16, input_dim=100, input_length=1, name="state")(sinput)
model = Model(inputs=[sinput], outputs=[sinput_em]) 
model.compile(loss={'state': 'sparse_categorical_crossentropy' },   optimizer='adam')

### Save model
from tensorflow.saved_model import simple_save
simple_save(K.get_session(),
            "SavedModel/testembedding",
            inputs={"state_input": sinput },
            outputs={'state': sinput_em})
# then run the script:
#  ~/.local/bin/freeze_graph  --input_saved_model_dir=SavedModel/testembedding --output_graph=frozenemb.pb  --output_node_names=state/embedding_lookup/Identity_1 --clear_devices

# try to load again:
frozen_graph="frozenemb.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
        )

# -> error in import_graph_def()

def serialize_graph(model):
    g = tf.graph_util.convert_variables_to_constants(
        tf.keras.backend.get_session(),
        tf.keras.backend.get_session().graph.as_graph_def(),
        #[n.name for n in tf.keras.backend.get_session().graph.as_graph_def().node],
        [t.op.name for t in model.outputs]
    )
    return g

sg = serialize_graph(model)

newg = tf.Graph()
with newg.as_default():
   tf.import_graph_def(sg) # -> error

# define the model, the one used now has an additional softmax layer, then:
saved_to_path = tf.keras.experimental.export(
      model, 'SavedModel/testembedding')

# import works fine
# Load the saved keras model back.
model_prime = tf.keras.experimental.load_from_saved_model(saved_path_model)
model_prime.summary()

# freezing:
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants

input_saved_model_dir = saved_to_path
output_graph_filename = "SaveFiles/testemb.pb"
output_node_names = "out/Softmax"  
input_binary = False
input_saver_def_path = False
restore_op_name = None
filename_tensor_name = None
clear_devices = False
input_meta_graph = False
checkpoint_path = None
input_graph_filename = None
saved_model_tags = tag_constants.SERVING

freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                            input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                            saved_model_tags)

# load again, -> ERROR: ValueError: Input 0 of node state/embedding_lookup was passed float from state/embeddings:0 incompatible with expected resource.
frozen_graph="SaveFiles/testemb.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
        )