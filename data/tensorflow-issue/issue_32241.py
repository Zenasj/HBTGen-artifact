from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

base_model = tf.keras.applications.NASNetMobile(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

output = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name="output")
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  output])
model.summary()

#verify inputs and outputs
print(model.input.op.name)
print(model.output.op.name) 
tf.keras.models.save_model(model, model_path, overwrite=True,save_format="h5")

NASNet_input
output/Identity

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.tools import freeze_graph

# unclear if strictly necessary but git issues imply it may be defensive to add these lines
tf.reset_default_graph()
K.clear_session()
K.set_learning_phase(0)

restored_model = tf.keras.models.load_model(model_path, compile=True)
print(restored_model.inputs)
print(restored_model.outputs)

restored_model.summary()

import datetime

output_model_name = model_name + ".pb"
output_model_path = "/tmp/" + output_model_name
save_dir = "./tmp_{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
tf.saved_model.simple_save(K.get_session(),
                           save_dir,
                           inputs={"input": restored_model.inputs[0]},
                           outputs={"output": restored_model.outputs[0]})

freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          restored_model.outputs[0].op.name,
                          None,
                          None,
                           output_model_path,
                          False,
                          "",
                          input_saved_model_dir=save_dir)

with tf.keras.backend.get_session() as session:
  graph = session.graph
  input_graph_def = graph.as_graph_def()
  
  # For demonstration purpose we show the first 15 ops the TF model
with graph.as_default() as g:
    tf.import_graph_def(input_graph_def, name='')
    
    ops = g.get_operations()
    for op in ops[0:15]:
        print('op name: {}, op type: "{}"'.format(op.name, op.type));
    for op in ops[::-1][0:15]:
        print('op name: {}, op type: "{}"'.format(op.name, op.type));

  

input_node_names = ['NASNet_input']
output_node_names = ['output/Sigmoid']
gdef = strip_unused_lib.strip_unused(
      input_graph_def = input_graph_def,
      input_node_names = input_node_names,
      output_node_names = output_node_names,
      placeholder_type_enum = dtypes.float32.as_datatype_enum)
  
with gfile.GFile(output_model_path, "wb") as f:
    f.write(gdef.SerializeToString())

# from tensorflow.python.framework.graph_util import convert_variables_to_constants


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



      
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in restored_model.outputs], 
                             clear_devices=True)

tf.train.write_graph(frozen_graph, "/tmp", model_name+".pb", as_text=False)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def freeze_graph_keras(net, model_dir):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """

    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    file_name = os.path.basename(model_dir).replace('.hdf5', '.pb')
    model_dir = os.path.dirname(model_dir)
    print(os.path.join(model_dir, file_name))
    with tf.keras.backend.get_session() as sess:
        tf.initialize_all_variables().run()
        frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in net.outputs])
        tf.train.write_graph(frozen_graph, model_dir,
                             file_name, as_text=False)
    print('All input nodes:', net.inputs)
    print('All output nodes:', net.outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="path.hdf5", help="Model folder to export")

    args = parser.parse_args()                    

    model = build_my_model((256,320,3), num_classes=4,
                    lr_init=1e-3, lr_decay=5e-4)

    freeze_graph_keras(model, args.model_dir)