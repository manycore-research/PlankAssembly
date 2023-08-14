# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import json
import os
import shapely

import numpy as np
import torch

from plankassembly.datasets.data_utils import quantize_values, add_noise


class LineDataset(torch.utils.data.Dataset):

    def __init__(self, root, info_files, token, cfg, augmentation=False):

        self.root = root
        self.info_files = info_files
        self.augmentation = augmentation
        self.token = token

        self.vocab_size = cfg.VOCAB_SIZE
        self.num_input_dof = cfg.NUM_INPUT_DOF
        self.max_input_length = cfg.MAX_INPUT_LENGTH
        self.max_output_length = cfg.MAX_OUTPUT_LENGTH
        self.num_bits = cfg.NUM_BITS

        self.aug_ratio = cfg.AUG_RATIO
        self.noise_ratio = cfg.NOISE_RATIO
        self.noise_length = cfg.NOISE_LENGTH

    def __len__(self):
        return len(self.info_files)

    def prepare_input_sequence(self, lines, views, types):
        # input
        input_value = quantize_values(np.array(lines), self.num_bits)
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
        input_coord = np.arange(len(input_value)) % self.num_input_dof

        # repeat for each token
        input_pos = np.repeat(input_pos, 4)
        input_view = np.repeat(input_view, 4)
        input_type = np.repeat(input_type, 4)

        # add stop token
        input_value = np.append(input_value, self.token.END)
        num_input = len(input_value)

        # add pad tokens
        pad_length = self.max_input_length - num_input

        input_value = np.pad(input_value, (0, pad_length-1), constant_values=self.token.PAD)
        input_pos = np.pad(input_pos, (0, pad_length))
        input_coord = np.pad(input_coord, (0, pad_length))
        input_view = np.pad(input_view, (0, pad_length))
        input_type = np.pad(input_type, (0, pad_length))
        input_mask = (input_value == self.token.PAD)

        inputs = {
            'input_value': input_value,
            'input_pos': input_pos,
            'input_coord': input_coord,
            'input_view': input_view,
            'input_type': input_type,
            'input_mask': input_mask
        }

        return inputs

    def prepare_output_sequence(self, planks, attach):
        # output
        value = quantize_values(planks, self.num_bits)

        # add stop token
        value = np.append(value, self.token.END)
        num_output = len(value)

        # add pad tokens
        value = np.pad(value, (0, self.max_output_length - num_output), constant_values=self.token.PAD)
        mask = (value == self.token.PAD)

        # label
        label = np.pad(attach, (0, self.max_output_length - len(attach)), constant_values=-1)

        label[label != -1] += self.vocab_size
        label[label == -1] = value[label == -1]

        outputs = {
            'output_value': value,
            'output_label': label,
            'output_mask': mask
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

        planks = np.array(info['coords']).flatten()
        attach = np.array(info['attach']).flatten()

        if self.augmentation and np.random.random() < self.aug_ratio:

            linestrings, views, types = add_noise(
                linestrings, views, types, self.noise_ratio, self.noise_length)

            lines = shapely.bounds(linestrings)

        inputs = self.prepare_input_sequence(lines, views, types)

        outputs = self.prepare_output_sequence(planks, attach)

        # construct batch data
        batch = {'name': name, **inputs, **outputs}

        return batch
