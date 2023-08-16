# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import json
import os

import numpy as np
import shapely
import torch

from polygen.data_utils import add_noise, quantize_values


class VertexDataset(torch.utils.data.Dataset):

    def __init__(self, root, info_files, token, cfg, augmentation=False):
        self.root = root
        self.info_files = info_files
        self.token = token
        self.augmentation = augmentation

        self.num_line_dof = cfg['num_line_dof']
        self.max_line_seq = cfg['max_line_seq']
        self.max_vert_seq = cfg['max_vert_seq']

        self.quantization_bits = cfg['quantization_bits']

        self.aug_ratio = cfg['aug_ratio']
        self.noise_ratio = cfg['noise_ratio']
        self.noise_length = cfg['noise_length']

    def __len__(self):
        return len(self.info_files)

    def prepare_input_sequence(self, lines, views, types):
        # input
        input_value = quantize_values(lines, self.quantization_bits)
        input_view = np.array(views)
        input_type = np.array(types)

        # sort lines by first by view, then by lines
        line_with_view = np.concatenate((input_value, input_view[..., np.newaxis]), axis=1)
        sort_inds = np.lexsort(line_with_view.T[[3, 1, 2, 0, 4]])

        input_value = input_value[sort_inds].flatten()
        input_view = input_view[sort_inds]
        input_type = input_type[sort_inds]

        # position
        _, counts = np.unique(input_view, return_counts=True)
        input_pos = np.concatenate([np.arange(count) for count in counts])

        # coordinate
        input_coord = np.arange(len(input_value)) % self.num_line_dof

        # repeat for each token
        input_pos = np.repeat(input_pos, 4)
        input_view = np.repeat(input_view, 4)
        input_type = np.repeat(input_type, 4)

        # add stop token
        input_value = np.append(input_value, self.token['END'])
        num_input = len(input_value)

        # add pad tokens
        pad_length = self.max_line_seq - num_input

        input_value = np.pad(input_value, (0, pad_length-1), constant_values=self.token['PAD'])
        input_pos = np.pad(input_pos, (0, pad_length))
        input_coord = np.pad(input_coord, (0, pad_length))
        input_view = np.pad(input_view, (0, pad_length))
        input_type = np.pad(input_type, (0, pad_length))
        input_mask = (input_value == self.token['PAD'])

        inputs = {
            'line_value': input_value,
            'line_pos': input_pos,
            'line_coord': input_coord,
            'line_view': input_view,
            'line_type': input_type,
            'line_mask': input_mask
        }

        return inputs

    def prepare_vertex_sequence(self, vertices):
        vertices = np.append(vertices, self.token['END'])
        vertices = np.pad(vertices, (0, self.max_vert_seq - len(vertices)), constant_values=self.token['PAD'])
        vertices_mask = (vertices == self.token['PAD'])

        outputs = {
            'vertex': vertices,
            'vertex_mask': vertices_mask,
        }

        return outputs

    def __getitem__(self, index):
        """ Load data for data i"""
        with open(os.path.join(self.root, self.info_files[index]), "r") as f:
            info = json.loads(f.read())

        name = info['name']
        svgs = info['svgs']
        
        linestrings = [shapely.from_geojson(svg) for svg in svgs]
    
        lines = np.array(info['lines'], dtype='float')
        views = np.array(info['views'], dtype='long')
        types = np.array(info['types'], dtype='long')

        vertices = np.array(info['vertices'], dtype='long').flatten()

        if self.augmentation and np.random.random() < self.aug_ratio:
            
            linestrings, views, types = add_noise(
                linestrings, views, types, self.noise_ratio, self.noise_length)

            lines = shapely.bounds(linestrings)

        inputs = self.prepare_input_sequence(lines, views, types)

        outputs = self.prepare_vertex_sequence(vertices)
        
        batch = {'name': name, **inputs, **outputs}

        return batch


class FaceDataset(torch.utils.data.Dataset):

    def __init__(self, root, info_files, token, cfg):
        self.root = root
        self.info_files = info_files
        self.token = token

        self.max_num_vert = cfg['max_num_vert']
        self.max_face_seq = cfg['max_face_seq']

    def __len__(self):
        return len(self.info_files)

    def __getitem__(self, index):
        """ Load data for data i"""
        with open(os.path.join(self.root, self.info_files[index]), "r") as f:
            info = json.loads(f.read())

        name = info['name']

        # load sequences
        vertices = np.array(info['vertices'], dtype='long')
        faces = np.array(info['faces'], dtype='long')

        # vertices
        num_vertices = len(vertices)

        vertices = np.pad(vertices, ((0, self.max_num_vert - num_vertices), (0, 0)))

        # vertex_mask
        vertices_mask = np.ones_like(vertices[..., 0], dtype=bool)
        vertices_mask[:num_vertices] = False

        # pad for special tokens
        vertices_mask = np.pad(vertices_mask, (3, 0), constant_values=0)

        # faces
        faces = np.pad(faces, (0, self.max_face_seq - len(faces)))
        faces_mask = (faces == self.token['PAD'])

        # construct batch data
        data = {
            'name': name,
            'vertex': vertices,
            'vertex_mask': vertices_mask,
            'face': faces,
            'face_mask': faces_mask,
        }
        return data


class TestDataset(VertexDataset):

    def __init__(self, root, info_files, token, cfg):
        super().__init__(root, info_files, token, cfg)

    def __getitem__(self, index):
        """ Load data for data i"""
        with open(os.path.join(self.root, self.info_files[index]), "r") as f:
            info = json.loads(f.read())

        name = info['name']

        lines = np.array(info['lines'], dtype='float')
        views = np.array(info['views'], dtype='long')
        types = np.array(info['types'], dtype='long')

        inputs = self.prepare_input_sequence(lines, views, types)

        batch = {'name': name, **inputs}

        return batch
