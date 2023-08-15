# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import json
import os
from io import BytesIO

import numpy as np
import shapely
import svgwrite
import torch
from cairosvg import svg2png
from PIL import Image


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


def collate_fn(data_list):
    """ Flatten a set of items from ScenesDataset into a batch.

    Pytorch dataloader has memory issues with nested and complex 
    data structures. This flattens the data into a dict of batched tensors.
    Frames are batched temporally as well (bxtxcxhxw)
    """

    keys = list(data_list[0].keys())
    if len(data_list[0]['frames']) > 0:
        frame_keys = list(data_list[0]['frames'][0].keys()) 
    else:
        frame_keys = []
    keys.remove('frames')

    out = {key:[] for key in keys+frame_keys}
    for data in data_list:
        for key in keys:
            out[key].append(data[key])

        for key in frame_keys:
            if torch.is_tensor(data['frames'][0][key]):
                out[key].append( torch.stack([frame[key] 
                                              for frame in data['frames']]) )
            else:
                # non tensor metadata may not exist for every frame
                # (ex: instance_file_name)
                out[key].append( [frame[key] if key in frame else None 
                                  for frame in data['frames']] )

    for key in out.keys():
        if torch.is_tensor(out[key][0]):
            out[key] = torch.stack(out[key])

    return out


def convert_svg_to_png(lines, types, image_size=128, line_width=0.01):
    dwg = svgwrite.Drawing()
    dwg.viewbox(-1, -1, 2, 2)
    dwg.defs.add(dwg.style(".vectorEffectClass {vector-effect: non-scaling-stroke;}"))

    for line, line_type in zip(lines, types):

        svg = svgwrite.shapes.Line(line[:2], line[2:], fill="none", class_='vectorEffectClass')
        svg.stroke("black", width=line_width)

        if line_type == 1:
            svg.dasharray([line_width*10, line_width*10])

        dwg.add(svg)

    image = svg2png(
        bytestring=dwg.tostring(),
        output_width=image_size,
        output_height=image_size,
        background_color='white',
    )

    return image


class LineDataset(torch.utils.data.Dataset):

    def __init__(self, root, info_files, token, 
                 transform=None, cfg=None, augmentation=False):

        self.root = root
        self.info_files = info_files
        self.token = token
        self.transform = transform
        self.augmentation = augmentation

        self.image_size = cfg.IMAGE_SIZE
        self.line_width = cfg.LINE_WIDTH

        self.vocab_size = cfg.VOCAB_SIZE
        self.max_output_length = cfg.MAX_OUTPUT_LENGTH
        self.num_bits = cfg.NUM_BITS

        self.aug_ratio = cfg.AUG_RATIO
        self.noise_ratio = cfg.NOISE_RATIO
        self.noise_length = cfg.NOISE_LENGTH

        self.poses = {
            0: np.linalg.inv([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),    # f
            1: np.linalg.inv([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),   # t
            2: np.linalg.inv([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])    # s
        }

    def __len__(self):
        return len(self.info_files)
    
    def prepare_input_frames(self, lines, views, types):
        frames = []

        for view_index in range(3):

            mask = views == view_index

            image_bytes = convert_svg_to_png(
                lines[mask], types[mask], self.image_size, self.line_width)
            stream = BytesIO(image_bytes)

            frame = {}
            frame['image'] = Image.open(stream)
            frame['intrinsics'] = np.array([
                [self.image_size / 2, 0, 0, self.image_size / 2],
                [0, self.image_size / 2, 0, self.image_size / 2],
                [0, 0, 0, 1]], dtype=np.float32)
            frame['pose'] = np.array(self.poses[view_index], dtype=np.float32)
            frames.append(frame)

        return frames

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
            'output_mask': mask,
            'output_label': label,
        }
        return outputs

    def __getitem__(self, index):
        """ Load data for data i"""
        with open(os.path.join(self.root, self.info_files[index]), "r") as f:
            info = json.loads(f.read())

        name = info['name']
        svgs = info['svgs']

        linestrings = [shapely.from_geojson(svg) for svg in svgs]

        lines, views, types = info['lines'], info['views'], info['types']

        planks = np.array(info['coords']).flatten()
        attach = np.array(info['attach']).flatten()

        if self.augmentation and np.random.random() < self.aug_ratio:

            linestrings, views, types = add_noise(
                linestrings, views, types, self.noise_ratio, self.noise_length)

            lines = shapely.bounds(linestrings)

        lines = np.array(lines, dtype='float')
        views = np.array(views, dtype='long')
        types = np.array(types, dtype='long')

        frames = self.prepare_input_frames(lines, views, types)

        # load sequences
        outputs = self.prepare_output_sequence(planks, attach)

        # construct batch data
        data = {'name': name, 'frames': frames, **outputs,}

        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == "__main__":

    dataset = LineDataset(
        root='data/train_data',
        split='data/splits/overfit.txt',
        max_num_seq=128,
        token={'END': 513, 'PAD': 514}
    )

    batch = dataset[0]
    for key, value in batch.items():
        print(key, value)
