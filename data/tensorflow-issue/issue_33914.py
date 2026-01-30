import tensorflow as tf
from tensorflow import keras

def freeze_session(session, model, keep_var_names=None, clear_devices=None):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = [out.op.name for out in model.outputs]
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names)#, freeze_var_names)
        return frozen_graph

def save_model(model, iteration):
    # K.clear_session()
    K.set_learning_phase(0)
    new_model = get_out_model(model)
    
    frozen_graph = freeze_session(K.get_session(), new_model)
    with tf.gfile.GFile('./model/model.pb', "wb") as f:
        f.write(frozen_graph.SerializeToString())
    # path = tf.train.write_graph(frozen_graph, './model', 'model' + str(iteration) + '.pb')

    
    converter = tf.lite.TFLiteConverter.from_frozen_graph('./model/model.pb',
                                                          input_arrays=new_model.input_names,
                                                          # output_arrays=['predictions/concat'])
                                                          output_arrays=['decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = set([tf.lite.OpsSet.SELECT_TF_OPS])
    converter.allow_custom_ops = True
    converter.drop_control_dependency = False
    tflite_model = converter.convert()
    with open("./model.tflite", 'wb') as tfile:
        tfile.write(tflite_model)
        
    K.set_learning_phase(1)

frozen_graph = optimize_for_inference_lib.optimize_for_inference(frozen_graph,
                                                                         [input],
                                                                         [output],
                                                                         tf.float32.as_datatype_enum)

lstm_model = tf.keras.Sequential(...)
converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
converter.experimental_new_converter = True
tflite_model = converter.convert()