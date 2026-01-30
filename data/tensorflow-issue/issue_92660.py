import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

class SpatialTransformer(layers.Layer):
    """3D Spatial Transformer using batched 2D warps and static shape enforcement."""
    def call(self, inputs):
        vol, flow = inputs  # vol: [B,D,H,W,C], flow: [B,D,H,W,3]

        # 1. Enforce static (non-zero) shapes to satisfy TOSA requirements
        #    (TOSA dialect expects all dims ≥ 1 and statically known) 
        vol  = tf.ensure_shape(vol,  [None, vol.shape[1], vol.shape[2], vol.shape[3], vol.shape[4]])
        flow = tf.ensure_shape(flow, [None, flow.shape[1], flow.shape[2], flow.shape[3], 3])          

        # 2. Flatten depth dimension into batch: [B,D,H,W,C] → [B*D,H,W,C]
        shape = tf.shape(vol)
        B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], vol.shape[4]
        vol_flat  = tf.reshape(vol,  tf.stack([B * D, H, W, C]))                                    
        flow_flat = tf.reshape(flow, tf.stack([B * D, H, W, 3]))                                     

        # 3. Perform a single batched 2D warp via dense_image_warp,
        #    avoiding tf.map_fn loops entirely :contentReference[oaicite:7]{index=7}
        moved_flat = tfa.image.dense_image_warp(vol_flat, flow_flat[..., :2])

        # 4. Restore original shape: [B*D,H,W,C] → [B,D,H,W,C]
        moved = tf.reshape(moved_flat, tf.stack([B, D, H, W, C]))                                     
        return moved

def conv_block(x, filters, convs=2, kernel_size=3, activation='relu'):
    for _ in range(convs):
        x = layers.Conv3D(filters, kernel_size, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.Activation(activation)(x)
    return x

def build_minimal_voxelmorph(inshape,
                             enc_features=(16, 32, 32, 32),
                             dec_features=(32, 32, 32, 32, 32, 16, 16)):
    moving = layers.Input(shape=(*inshape, 1), name='moving')
    fixed  = layers.Input(shape=(*inshape, 1), name='fixed')
    x = layers.Concatenate(axis=-1)([moving, fixed])

    skips = []
    for f in enc_features:
        x = conv_block(x, f)
        skips.append(x)
        x = layers.MaxPool3D(2)(x)

    x = conv_block(x, enc_features[-1] * 2)

    for f, skip in zip(dec_features, reversed(skips)):
        x = layers.UpSampling3D(2)(x)
        x = layers.Concatenate(axis=-1)([x, skip])
        x = conv_block(x, f)

    flow  = layers.Conv3D(3, 3, padding='same', name='flow')(x)
    moved = SpatialTransformer(name='moved')([moving, flow])

    return models.Model(inputs=[moving, fixed],
                        outputs=[moved, flow],
                        name='VoxelmorphMinimalFlatten')

# Instantiate model for a 128³ volume                        
model = build_minimal_voxelmorph((128, 128, 128))
model.summary()

save_path = os.path.join(os.getcwd(), "model/simple/")
tf.saved_model.save(model, save_path) 

@tf.function
def infer(moving, fixed):
    return model([moving, fixed])

inp0, inp1 = model.inputs

concrete_func = infer.get_concrete_function(
    moving=tf.TensorSpec(shape=inp0.shape, dtype=inp0.dtype, name=inp0.name.split(':')[0]),
    fixed =tf.TensorSpec(shape=inp1.shape, dtype=inp1.dtype, name=inp1.name.split(':')[0])
)

frozen_func = convert_variables_to_constants_v2(concrete_func)
tf.io.write_graph(
    graph_or_graph_def=frozen_func.graph,
    logdir=os.getcwd(),
    name="output/frozen_graph.pbtxt",
    as_text=True
)

print("Frozen graph:")
with tf.io.gfile.GFile("output/frozen_graph.pbtxt", "r") as f:
    frozen_graph = f.read()
    print(frozen_graph)