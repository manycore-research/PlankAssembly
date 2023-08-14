# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import json
import os

import numpy as np
import shapely
import torch

from plankassembly.datasets.data_utils import add_noise, quantize_values


class Sideface():
    def __init__(self, linestring, line_width, line_type):
        self.linestring = linestring
        self.line_width = line_width
        self.line_type = line_type

    def to_polygon(self):
        return shapely.buffer(self.linestring, self.line_width / 2, cap_style="flat")


def parse_sideface_from_polygons(polygons, max_thickness):

    lines = []
    for polygon in polygons:
        bounds = shapely.bounds(polygon).reshape(-1, 2)
        diffs = np.diff(bounds, axis=0).flatten()
        center = np.mean(bounds, 0)

        if diffs[1] < max_thickness:
            line = shapely.linestrings([bounds[0][0], bounds[1][0]], [center[1], center[1]])
            lines.append(Sideface(line, diffs[1], 1))

        if diffs[0] < max_thickness:
            line = shapely.linestrings([center[0], center[0]], [bounds[0][1], bounds[1][1]])
            lines.append(Sideface(line, diffs[0], 0))

    return lines


def merge_colinaer_sidefaces(lines, merge_tolerance, min_thickness):

    merged_lines = [lines[0], ]

    for query_line in lines[1:]:

        tree = shapely.STRtree([line.linestring for line in merged_lines])
        indices = tree.query(query_line.linestring, predicate='intersects')

        colinear_indices = []

        for index in np.sort(indices):
            # find colinear case and merge
            coords = shapely.get_coordinates([query_line.linestring, merged_lines[index].linestring])

            if ((np.max(coords[:, 0]) - np.min(coords[:, 0])) < merge_tolerance or
                (np.max(coords[:, 1]) - np.min(coords[:, 1])) < merge_tolerance) \
                and np.abs(query_line.line_width - merged_lines[index].line_width) < merge_tolerance \
                and query_line.line_type == merged_lines[index].line_type:
                colinear_indices.append(index)

        if len(colinear_indices) > 0:
            # merge colinear lines
            multilinestrings = shapely.multilinestrings(
                [query_line.linestring, ] + [merged_lines[i].linestring for i in colinear_indices])
            bounds = shapely.bounds(multilinestrings)
            bounds = np.array(bounds).reshape(2, 2)
            linestring = shapely.linestrings(*bounds.T)
            query_line = Sideface(linestring, query_line.line_width, query_line.line_type)

            # remove merged lines
            for i in reversed(colinear_indices):
                merged_lines.pop(i)

        # append new line
        merged_lines.append(query_line)

    merged_lines = [line.to_polygon() for line in merged_lines if line.line_width >= min_thickness]

    return merged_lines


class SidefaceDataset(torch.utils.data.Dataset):

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

        self.max_thickness = cfg.MAX_THICKNESS / cfg.SCALE
        self.min_thickness = cfg.MIN_THICKNESS / cfg.SCALE
        self.merge_tolerance = cfg.MERGE_TOLERANCE / cfg.SCALE

    def __len__(self):
        return len(self.info_files)

    def extract_sideface(self, linestrings, views):

        sidefaces = []
        faceviews = []

        for view_index in range(3):

            line = [l_i for l_i, v_i in zip(linestrings, views) if v_i == view_index]

            if len(line) == 0:
                continue

            polygon = shapely.get_parts(shapely.polygonize(line))

            sideface = parse_sideface_from_polygons(polygon, self.max_thickness)

            if len(sideface) == 0:
                continue

            merged_sideface = merge_colinaer_sidefaces(sideface, self.merge_tolerance, self.min_thickness)

            sidefaces.extend(merged_sideface)
            faceviews.extend([view_index, ] * len(merged_sideface))

        sidefaces = shapely.bounds(sidefaces)

        return sidefaces, faceviews

    def prepare_input_sequence(self, faces, views):
        # input
        input_value = quantize_values(np.array(faces), self.num_bits)
        input_view = np.array(views, dtype='long')
            
        if len(faces) != 0:
            # sort faces by first by view, then by lines
            face_with_view = np.concatenate((input_value, input_view[..., np.newaxis]), axis=1)
            sort_inds = np.lexsort(face_with_view.T[[3, 1, 2, 0, 4]])

            input_value = input_value[sort_inds].flatten()
            input_view = input_view[sort_inds]

            # position
            _, counts = np.unique(input_view, return_counts=True)
            input_pos = np.concatenate([np.arange(count) for count in counts])

            # coordinate
            input_coord = np.arange(len(input_value)) % self.num_input_dof

            # repeat for each token
            input_pos = np.repeat(input_pos, 4)
            input_view = np.repeat(input_view, 4)

        else:
            # deal with empty sidefaces
            input_pos = np.zeros_like(input_view, dtype='long')
            input_coord = np.zeros_like(input_view, dtype='long')

        # add stop token
        input_value = np.append(input_value, self.token.END)
        num_input = len(input_value)

        # add pad tokens
        pad_length = self.max_input_length - num_input

        input_value = np.pad(input_value, (0, pad_length-1), constant_values=self.token.PAD)
        input_pos = np.pad(input_pos, (0, pad_length))
        input_coord = np.pad(input_coord, (0, pad_length))
        input_view = np.pad(input_view, (0, pad_length))
        input_mask = (input_value == self.token.PAD)

        inputs = {
            'input_value': input_value,
            'input_pos': input_pos,
            'input_coord': input_coord,
            'input_view': input_view,
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

        views = np.array(info['views'], dtype='long')
        types = np.array(info['types'], dtype='long')

        planks = np.array(info['coords']).flatten()
        attach = np.array(info['attach']).flatten()

        sidefaces, faceviews = [], []

        if self.augmentation and np.random.random() < self.aug_ratio:

            noisy_linestrings, noisy_views, _ = add_noise(
                linestrings, views, types, self.noise_ratio, self.noise_length)

            sidefaces, faceviews = self.extract_sideface(noisy_linestrings, noisy_views)

        # detect degenerated case
        if len(sidefaces) == 0:
            linestrings = [shapely.from_geojson(svg) for svg in svgs]
            views = np.array(info['views'], dtype='long')

            sidefaces, faceviews = self.extract_sideface(linestrings, views)

        inputs = self.prepare_input_sequence(sidefaces, faceviews)

        outputs = self.prepare_output_sequence(planks, attach)

        # construct batch data
        batch = {'name': name, **inputs, **outputs}

        return batch
