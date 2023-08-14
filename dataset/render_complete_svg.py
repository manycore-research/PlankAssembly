# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
"""
render complete vectorized svgs
"""
import argparse
import json
import os

from tqdm.contrib.concurrent import process_map

from data_utils import *


def post_process(lines, line_types):

    lines, line_types = split_lines_on_crossing_points(lines, line_types)

    lines, line_types = split_lines_on_endpoints(lines, line_types)

    lines, line_types = remove_overlapping_lines(lines, line_types)

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
    parser.add_argument('--data_type', type=str, default="complete",
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
