import torch.nn as nn

mesh = load_objs_as_meshes([os.path.join(path, 'mesh.obj')], device=device)

criterion = torch.nn.MSELoss()

deform_verts = torch.full(mesh.verts_packed().shape, 0.0, dtype=torch.float32, device=meta.device, requires_grad=True)
mesh = mesh.offset_verts(deform_verts)

projection = project_mesh(smpl_mesh, angle).to(device)[0, :, :, 0]  # gives the projected image at the required angle
loss = criterion(torch.flatten(projection), torch.flatten(ground_truth))

loss.backward()

import os
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader
)


path = "NOMO_preprocess/data"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Load any mesh and set req_grad True for vertices
verts, faces_idx, _ = load_obj(os.path.join(path, 'male.obj'))
verts.requires_grad = True
faces = faces_idx.verts_idx
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(device))
smpl_mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures
)

criterion = torch.nn.MSELoss()

# Project the mesh at any angle
R, T = look_at_view_transform(1.75, -45, 0, up=((0, 1, 0),), at=((0, -0.25, 0),))
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardFlatShader(device=device, lights=lights)
)
verts = smpl_mesh.verts_list()[0]
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(device))
smpl_mesh.textures = textures
smpl_mesh.textures._num_faces_per_mesh = smpl_mesh._num_faces_per_mesh.tolist()
smpl_mesh.textures._num_verts_per_mesh = smpl_mesh._num_verts_per_mesh.tolist()

projection = renderer(smpl_mesh)[0, :, :, 0]

ground_truth = torch.ones(projection.size(), device=device)
loss = criterion(projection, ground_truth)
loss.backward()