# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from mesh_utils import build_mesh
from plankassembly.data import dequantize_values


def main():
    filenames = os.listdir(os.path.join(args.exp_path, 'pred_jsons'))
    
    for filename in tqdm(filenames):
        
        if not filename.endswith('.json'):
            continue

        with open(os.path.join(args.exp_path, 'pred_jsons', filename), 'r') as f:
            infos = json.load(f)

        coords = dequantize_values(np.array(infos['prediction']))

        mesh = build_mesh(coords)
        mesh.export(os.path.join(args.exp_path, 'pred_meshes', filename.replace('.json', '.stl')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default="lightning_logs/version_X",
                        help='dataset path.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.exp_path, 'pred_meshes'), exist_ok=True)

    main()