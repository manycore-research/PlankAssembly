# from Atlas: (https://github.com/magicleap/atlas)

import numpy as np
import torch


class Compose(object):
    """ Apply a list of transforms sequentially"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ToTensor(object):
    """ Convert to torch tensors"""
    def __call__(self, data):
        for frame in data['frames']:
            image = np.array(frame['image'])
            frame['image'] = torch.as_tensor(image).float().permute(2, 0, 1)
            frame['intrinsics'] = torch.as_tensor(frame['intrinsics'])
            frame['pose'] = torch.as_tensor(frame['pose'])
        for key in data.keys():
            if key.startswith('output'):
                data[key] = torch.as_tensor(data[key])
        return data


class IntrinsicsPoseToProjection(object):
    """ Convert intrinsics and extrinsics matrices to a single projection matrix"""
    def __call__(self, data):
        for frame in data['frames']:
            intrinsics = frame.pop('intrinsics')
            pose = frame.pop('pose')
            frame['projection'] = intrinsics @ pose.inverse()
        return data
