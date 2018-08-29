"""
Pytorch dataset to load the data in the format presented in the paper.
The dataset returns images and viewpoints in a scene.
A scene has multiple images and each image has an associated viewpoint,
which is comprised of the tuple(w, y, p) where:
w -> (x, y, z) coordinates of the camera in scene with fixed frame of reference
y -> yaw
p-> pitch
"""

import os, sys, io, collections
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transformers import ToTensor


Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

def transform_viewpoints(v):
    """
    v: viewpoint
    Transforms the viewpoint into suitable representation
    v_hat = [w, cos(y), sin(y), cos(p), sin(p)]
    """

    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    _v = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(_v, dim=-1)

    return v_hat


class ShepardMetzler(Dataset):
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, f'{idx}.pt')
        data = torch.load(scene_path)

        byte_to_tensor = lambda x: ToTensor()(Image.open(io.BytesIO(x)))

        images = torch.stack([byte_to_tensor(frame) for frame in data.frames])
        viewpoints = torch.from_numpy(data.cameras)
        viewpoints = viewpoints.view(-1, 5)

        if self.transform:
            images = self.transform(images)

        if self.target_transform:
            images = self.target_transform(images)

        return images, viewpoints
