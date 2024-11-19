# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import json
import os
from io import BytesIO

import jsonlines
import numpy as np
import shapely
import svgwrite
import torch
from cairosvg import svg2png
from PIL import Image
from torchvision import transforms

from plankassembly.datasets.data_utils import add_noise, quantize_values

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def convert_svg_to_png(lines, types, image_size=224, line_width=0.01):
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


class ImageDataset(torch.utils.data.Dataset):

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
        self.image_size = cfg.IMAGE_SIZE
        self.view_size = self.image_size // 2
        self.line_width = cfg.get('LINE_WIDTH', 0.01)

        image_mean = cfg.IMAGE_MEAN if cfg.get('IMAGE_MEAN') else IMAGENET_DEFAULT_MEAN
        image_std = cfg.IMAGE_STD if cfg.get('IMAGE_STD') else IMAGENET_DEFAULT_STD

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

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

    def prepare_input_frames(self, lines, views, types):
        
        canvas = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        for view_index in range(3):

            mask = views == view_index

            image_bytes = convert_svg_to_png(
                lines[mask], types[mask], self.view_size, self.line_width)
            stream = BytesIO(image_bytes)
            image = Image.open(stream)

            if view_index == 0:
                canvas.paste(image, (0, self.view_size))
            elif view_index == 1:
                canvas.paste(image, (0, 0))
            else:
                canvas.paste(image, (self.view_size, self.view_size))

        return canvas

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
        
        lines = np.array(lines, dtype='float')
        views = np.array(views, dtype='long')
        types = np.array(types, dtype='long')

        image = self.prepare_input_frames(lines, views, types)
        image = self.transform(image)

        outputs = self.prepare_output_sequence(planks, attach)

        # construct batch data
        batch = {'name': name, 'image': image, **outputs}

        return batch
