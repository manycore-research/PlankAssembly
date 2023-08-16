# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import os

import shapely
import numpy as np


def parse_splits_list(splits):
    """ Returns a list of info_file paths
    Args:
        splits (list of strings): each item is a path to a .json file 
            or a path to a .txt file containing a list of paths to .json's.
    """

    if isinstance(splits, str):
        splits = splits.split()
    info_files = []
    for split in splits:
        ext = os.path.splitext(split)[1]
        if ext=='.json':
            info_files.append(split)
        elif ext=='.txt':
            info_files += [info_file.rstrip() for info_file in open(split, 'r')]
        else:
            raise NotImplementedError('%s not a valid info_file type'%split)
    return info_files


def add_noise(lines, views, types, noise_ratio, noise_length):
    num_select = np.random.randint(low=1, high=np.ceil(len(lines) * noise_ratio) + 1)

    indices = np.random.choice(len(lines), num_select, replace=False)

    for index in indices:

        if np.random.random() > 0.5:
            # delete
            lines[index] = None

        else:
            line = lines[index]
            length = shapely.length(line)

            noise = np.random.random() * noise_length
            noise = np.round(noise, 3)

            if length <= noise:
                # delete the line if it is too short
                lines[index] = None

            else:
                if np.random.random() > 0.5:
                    points = shapely.line_interpolate_point(line, [0, -noise])
                    points = np.concatenate([shapely.get_coordinates(point) for point in points.tolist()])
                    line = shapely.linestrings(points)
                else:
                    points = shapely.line_interpolate_point(line, [noise, length])
                    points = np.concatenate([shapely.get_coordinates(point) for point in points.tolist()])
                    line = shapely.linestrings(points)

                lines[index] = line

    noisy_lines, noisy_views, noisy_types = [], [], []
    for line, view, line_type in zip(lines, views, types):

        if line is None:
            continue

        noisy_lines.append(line)
        noisy_views.append(view)
        noisy_types.append(line_type)

    return noisy_lines, noisy_views, noisy_types


def quantize_values(verts, n_bits=9):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    min_range = -1
    max_range = 1
    range_quantize = 2**n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    return verts_quantize.astype('long')


def dequantize_values(quantized_verts, n_bits=9):
    """Convert dicretized vertices in [0, n_bits**2 - 1] to continuous values in [-1.0, -1.0]."""
    min_range = -1
    max_range = 1
    range_quantize = 2**n_bits - 1
    verts = quantized_verts * (max_range - min_range) / range_quantize + min_range
    return verts.astype('float')
