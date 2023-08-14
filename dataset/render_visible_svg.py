# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
"""
render visible vectorized svgs
"""
import argparse
import json
import os

import numpy as np
import shapely
from tqdm.contrib.concurrent import process_map

from data_utils import *


def filter_hidden_lines(lines, line_types):
    visible_lines = [line for line, line_type in zip(lines, line_types) if line_type == 0]
    line_types = [0, ] * len(visible_lines)
    return visible_lines, line_types
    

def merge_degenerated_lines(lines):

    while True:
        endpoints = shapely.multipoints(np.concatenate([np.array(line.coords) for line in lines]))
        endpoints = shapely.get_parts(shapely.extract_unique_points(endpoints)).tolist()
        
        tree = shapely.STRtree(endpoints)
        line_indices, point_indices = tree.query(lines, predicate='touches')

        unique_point_indices, counts = np.unique(point_indices, return_counts=True)

        if np.all(counts != 2):
            break

        # colinar
        done = True
        for point_index in unique_point_indices[counts == 2]:
            i, j = line_indices[point_indices == point_index]
            
            if lines[i] is None or lines[j] is None:
                done = False

            line_i, line_j = lines[i], lines[j]

            # find colinear case and merge
            coords = shapely.get_coordinates([line_i, line_j])

            if len(np.unique(coords[:, 0])) == 1 or len(np.unique(coords[:, 1])) == 1:
                merged_line = shapely.multilinestrings([line_i, line_j])
                bounds = shapely.bounds(merged_line)
                bounds = np.array(bounds).reshape(2, 2)
                line = shapely.linestrings(*bounds.T)

                # remove merged lines
                lines[i] = None
                lines[j] = None
                lines.append(line)

        lines = [line for line in lines if line is not None]

        if done:
            break

    return lines


def post_process(lines, line_types):

    lines, line_types = filter_hidden_lines(lines, line_types)

    lines, line_types = split_lines_on_crossing_points(lines, line_types)

    lines, line_types = split_lines_on_endpoints(lines, line_types)

    lines, line_types = remove_overlapping_lines(lines, line_types)

    lines = merge_degenerated_lines(lines)

    return lines, line_types


def render_three_views(name):
    try:
        with open(os.path.join(args.root, "model", f"{name}.json"), "r") as f:
            annos = json.loads(f.read())

        shape = build(annos['planks'])

        for view in VPS:

            lines, line_types = project(shape, view, args.decimals)

            lines, line_types = post_process(lines, line_types)

            render_svg(lines, line_types, view, name, args)

    except Exception as re:
        print(f'{name} failed, due to: {re}')


def main(args):
    info_files = parse_splits_list([
        os.path.join(args.root, 'splits', 'train.txt'),
        os.path.join(args.root, 'splits', 'valid.txt'),
        os.path.join(args.root, 'splits', 'test.txt')])

    names = [info_file.split('.')[0] for info_file in info_files]

    process_map(
        render_three_views, names,
        max_workers=args.max_workers, chunksize=args.chunksize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', metavar="DIR", default="data",
                        help='dataset source root.')
    parser.add_argument('--data_type', type=str, default="visible",
                        help='data type.')
    parser.add_argument('--name', type=str, default="",
                        help='data name.')
    parser.add_argument("--max_workers", default=16, type=int,
                        help="maximum number of workers")
    parser.add_argument("--chunksize", default=16, type=int,
                        help="chunk size")
    parser.add_argument('--line_width', type=str, default=0.5,
                        help='svg line width.')
    parser.add_argument('--decimals', type=int, default=3,
                        help='number of decimals.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.root, 'data', args.data_type, 'svgs'), exist_ok=True)

    if args.name:
        render_three_views(args.name, args)
    else:
        main(args)
