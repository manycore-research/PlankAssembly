# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os

import numpy as np
import shapely
from svgpathtools import svg2paths
from tqdm.contrib.concurrent import process_map

from data_utils import parse_splits_list


def parse_svg(name, view):
    # load svg
    paths, attributes = svg2paths(os.path.join(args.data_path, 'data', args.data_type, 'svgs', f'{name}_{view}.svg'))

    # parse svg paths
    lines = []
    line_types = []

    for p, attribute in zip(paths, attributes):
        # missing line
        if attribute['stroke'] == 'red':
            continue

        line = shapely.linestrings([[p.start.real, p.start.imag], [p.end.real, p.end.imag]])

        line_type = int('stroke-dasharray' in attribute)

        lines.append(line)
        line_types.append(line_type)

    return lines, line_types


def prepare_annotation(name):
    # load ground truths
    with open(os.path.join(args.data_path, "model", f"{name}.json"), "r") as f:
        infos = json.loads(f.read())

    svgs, types, views = [], [], []

    for v_i, view in enumerate(['f', 't', 's']):

        svg, line_type = parse_svg(name, view)

        svgs.extend(svg)
        types.extend(line_type)
        views.extend([v_i, ] * len(svg))

    lines = [shapely.bounds(line).tolist() for line in svgs]

    svgs = [shapely.to_geojson(line) for line in svgs]

    coords = np.array(infos['planks']) / args.scale
    coords = np.round(coords, decimals=args.decimals).tolist()

    with open(os.path.join(args.data_path, 'data', args.data_type, 'infos', f'{name}.json'), 'w') as f:
        json.dump({
            "name": name,
            # 2D inputs
            "lines": lines,
            "views": views,
            "types": types,
            "svgs": svgs,
            # 3D shape program
            "coords": coords,
            "attach": infos['attach']
        }, f)


def main(args):
    if 'noise' in args.data_type:
        info_files = parse_splits_list([
            os.path.join(args.data_path, 'splits', 'test.txt')])
    else:
        info_files = parse_splits_list([
            os.path.join(args.data_path, 'splits', 'train.txt'),
            os.path.join(args.data_path, 'splits', 'valid.txt'),
            os.path.join(args.data_path, 'splits', 'test.txt')])

    names = [info_file.split('.')[0] for info_file in info_files]

    process_map(
        prepare_annotation, names,
        max_workers=args.max_workers, chunksize=args.chunksize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', metavar="DIR", default="data",
                        help='data path.')
    parser.add_argument('--data_type', type=str, default="complete",
                        help='data type (complete/noise_x/visible).')
    parser.add_argument('--name', type=str, default="",
                        help='data name.')
    parser.add_argument("--max_workers", default=16, type=int,
                        help="maximum number of workers")
    parser.add_argument("--scale", default=1280, type=float,
                        help="object scale")
    parser.add_argument("--chunksize", default=16, type=int,
                        help="chunk size")
    parser.add_argument('--decimals', type=int, default=3,
                        help='svg line width.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_path, 'data', args.data_type, 'infos'), exist_ok=True)

    if args.name:
        prepare_annotation(args.name, args)
    else:
        main(args)
