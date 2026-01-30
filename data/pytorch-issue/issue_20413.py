import numpy as np
import trimesh

import torch
from torch.utils.tensorboard import SummaryWriter

sample_mesh = 'https://storage.googleapis.com/tensorflow-graphics/tensorboard/test_data/ShortDance07_a175_00001.ply'
log_dir = 'runs/torch'
batch_size = 1

# Camera and scene configuration.
config_dict = {
    'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
    'lights': [
        {
            'cls': 'AmbientLight',
            'color': '#ffffff',
            'intensity': 0.75,
        }, {
            'cls': 'DirectionalLight',
            'color': '#ffffff',
            'intensity': 0.75,
            'position': [0, -1, 2],
        }],
    'material': {
        'cls': 'MeshStandardMaterial',
        'roughness': 1,
        'metalness': 0
    }
}

# Read all sample PLY files.
mesh = trimesh.load_remote(sample_mesh)
vertices = np.array(mesh.vertices)
# Currently only supports RGB colors.
colors = np.array(mesh.visual.vertex_colors[:, :3])
faces = np.array(mesh.faces)

# Add batch dimension, so our data will be of shape BxNxC.
vertices = np.expand_dims(vertices, 0)
colors = np.expand_dims(colors, 0)
faces = np.expand_dims(faces, 0)

# Create data placeholders of the same shape as data itself.
vertices_tensor = torch.as_tensor(vertices)
faces_tensor = torch.as_tensor(faces)
colors_tensor = torch.as_tensor(colors)

writer = SummaryWriter(log_dir)

writer.add_mesh('mesh_color_tensor', vertices=vertices_tensor, faces=faces_tensor,
                colors=colors_tensor, config_dict=config_dict)

writer.close()