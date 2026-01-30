from tensorflow import keras

import tensorflow as tf
import json

def sample_network(input_layer):
    s2d = tf.nn.space_to_depth(input_layer, block_size=2, name="Space2Depth")
    d2s = tf.nn.depth_to_space(s2d, block_size=2, name="Depth2Space")
    return d2s

if __name__ == "__main__":
    input_net = tf.keras.Input(shape=(64, 64, 3), dtype=tf.float32)
    output = sample_network(input_net)
    model = tf.keras.Model(inputs=input_net, outputs=output)
    model_json = json.loads(model.to_json())
    with open("github_issue.json", "w") as f:
        json.dump(model_json, f, indent=2)

def sample_network(input_layer):
    with tf.name_scope("Space2Depth"):
        s2d = tf.nn.space_to_depth(input_layer, block_size=2, name="s2d1")
    with tf.name_scope("Depth2Space"):
        d2s = tf.nn.depth_to_space(s2d, block_size=2, name="d2s1")
    print(s2d.name, d2s.name)  # print out: tf.nn.space_to_depth/SpaceToDepth:0 tf.nn.depth_to_space/DepthToSpace:0
    return d2s

import tensorflow as tf
import json

def sample_network(input_layer):
    s2d = tf.nn.space_to_depth(input_layer, block_size=2, name="Space2Depth")
    mul = tf.multiply(s2d, 10.0, name="Multiplication")
    d2s = tf.nn.depth_to_space(mul, block_size=2, name="Depth2Space")
    print(s2d.name, d2s.name, mul.name) 
    return d2s


if __name__ == "__main__":
    # Disable eager mode
    tf.compat.v1.disable_eager_execution()

    input_net = tf.keras.Input(shape=(64, 64, 3), dtype=tf.float32, name="inputLayer")
    output = sample_network(input_net)
    model = tf.keras.Model(inputs=input_net, outputs=output)
    for layer in model.layers:
        print(layer.name)