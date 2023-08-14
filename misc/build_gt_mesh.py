# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os

from tqdm import tqdm

from mesh_utils import build_mesh


def main():
    with open(os.path.join(args.data_path, 'splits', 'test.txt'), 'r') as f:
        names = [line.rstrip().split('.')[0] for line in f]

    for name in tqdm(names):

        with open(os.path.join(args.data_path, 'data/complete', 'infos', f'{name}.json')) as f:
            infos = json.load(f)

        mesh = build_mesh(infos['coords'], transparent=True)
        mesh.export(os.path.join(args.data_path, 'mesh', f'{name}.stl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data",
                        help='dataset path.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_path, 'mesh'), exist_ok=True)
    
    main()
