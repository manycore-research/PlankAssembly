# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
"""
render noise vectorized svgs
"""
import argparse
import json
import os

import numpy as np
import shapely
import svgwrite
from tqdm.contrib.concurrent import process_map

from data_utils import *


def add_noises(lines):

    num_select = int(np.ceil(len(lines) * args.noise_ratio))

    indices = np.random.permutation(len(lines))
    indices = indices[:num_select]

    noise_types = np.zeros_like(lines, dtype=int).tolist()

    for index in indices:
        
        if np.random.random() > 0.5:
            # delete
            noise_types[index] = 1

        else:
            line = lines[index]
            length = shapely.length(line)

            noise = np.random.rand() * args.noise_length
            noise = np.round(noise, 3)

            if length <= noise:
                # delete the line if it is too short
                noise_types[index] = 1

            else:
                if np.random.rand() > 0.5:
                    points = shapely.line_interpolate_point(line, [0, -noise])
                    points = np.concatenate([shapely.get_coordinates(point) for point in points.tolist()])
                    line = shapely.linestrings(points)
                else:
                    points = shapely.line_interpolate_point(line, [noise, length])
                    points = np.concatenate([shapely.get_coordinates(point) for point in points.tolist()])
                    line = shapely.linestrings(points)

                lines[index] = line
                noise_types[index] = 2

    return lines, noise_types


def post_process(lines, line_types):

    lines, line_types = split_lines_on_crossing_points(lines, line_types)

    lines, line_types = split_lines_on_endpoints(lines, line_types)

    lines, line_types = remove_overlapping_lines(lines, line_types)

    lines, noise_types = add_noises(lines)

    return lines, line_types, noise_types


def render_svg(lines, line_types, noise_types, view, name):
    dwg = svgwrite.Drawing(os.path.join(args.root, 'data', args.data_type, 'svgs', f'{name}_{view}.svg'))
    dwg.viewbox(-1, -1, 2, 2)
    dwg.defs.add(dwg.style(".vectorEffectClass {vector-effect: non-scaling-stroke;}"))

    for line, line_type, noise_type in zip(lines, line_types, noise_types):

        endpoints = shapely.get_coordinates(line)
        svg = svgwrite.shapes.Line(endpoints[0], endpoints[1], fill="none", class_='vectorEffectClass')

        if line_type == 1:
            svg.dasharray([args.line_width*10, args.line_width*10])

        if noise_type == 0:
            # no noise
            svg.stroke("black", width=args.line_width)

        elif noise_type == 1:
            # missing
            svg.stroke("red", width=args.line_width)

        elif noise_type == 2:
            # shorten
            svg.stroke("blue", width=args.line_width)

        dwg.add(svg)

    dwg.save(pretty=True)


def render_three_views(name):
    try:
        index, name = name

        np.random.seed(index)

        with open(os.path.join(args.root, "model", f"{name}.json"), "r") as f:
            annos = json.loads(f.read())

        shape = build(annos['planks'])

        for view in VPS:
            
            lines, line_types = project(shape, view, args.decimals)

            lines, line_types, noise_types = post_process(lines, line_types)

            render_svg(lines, line_types, noise_types, view, name)

    except Exception as re:
        print(f'{name} failed, due to: {re}')


def main(args):
    info_files = parse_splits_list([
        os.path.join(args.root, 'splits', 'test.txt')])

    names = [(index, info_file.split('.')[0]) for index, info_file in enumerate(info_files)]

    process_map(
        render_three_views, names,
        max_workers=args.max_workers, chunksize=args.chunksize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', metavar="DIR", default="data",
                        help='dataset source root.')
    parser.add_argument('--data_type', type=str, default="noise_05",
                        help='data type.')
    parser.add_argument('--noise_ratio', type=float, default=0.05,
                        help='noise level.')
    parser.add_argument('--noise_length', type=float, default=0.02,
                        help='noise segment.')
    parser.add_argument('--name', type=str, default="",
                        help='data name.')
    parser.add_argument("--max_workers", default=16, type=int,
                        help="maximum number of workers")
    parser.add_argument("--chunksize", default=16, type=int,
                        help="chunk size")
    parser.add_argument('--line_width', type=str, default=0.5,
                        help='svg line width.')
    parser.add_argument('--decimals', type=int, default=3,
                        help='svg line width.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.root, 'data', args.data_type, 'svgs'), exist_ok=True)

    if args.name:
        render_three_views((0, args.name), args)
    else:
        main(args)
