# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os

from html4vision import Col, imagetable
from tqdm import tqdm
import numpy as np

from mesh_utils import build_mesh


def main():
    with open(os.path.join(args.data_path, 'splits', 'test.txt'), 'r') as f:
        test_names = [line.rstrip().split('.')[0] for line in f]

    np.random.shuffle(test_names)

    names = []
    precisions = []
    recalls = []
    fmeasures = []

    for name in tqdm(test_names[:300]):

        if not os.path.exists(os.path.join(args.exp_path, 'pred_jsons', f'{name}.json')):
            continue
        
        names.append(name)

        with open(os.path.join(args.exp_path, 'pred_jsons', f'{name}.json')) as f:
            results = json.load(f)

        mesh = build_mesh(results['prediction'], transparent=True)
        mesh.export(os.path.join(args.exp_path, 'pred_mesh', f'{name}.glb'))

        mesh = build_mesh(results['groundtruth'], transparent=True)
        mesh.export(os.path.join(args.exp_path, 'gt_mesh', f'{name}.glb'))

        with open(os.path.join(args.exp_path, 'metrics.json')) as f:
            metrics = json.load(f)

        precisions.append(round(metrics[name]['precision'], 4) * 100)
        recalls.append(round(metrics[name]['recall'], 4) * 100)
        fmeasures.append(round(metrics[name]['fmeasure'], 4) * 100)

    columns = [
        Col('text', 'ID', names),
        Col('img', 'Front', [os.path.join('svgs', f'{name}_f.svg') for name in names]),
        Col('img', 'Top', [os.path.join('svgs', f'{name}_t.svg') for name in names]),
        Col('img', 'Side', [os.path.join('svgs', f'{name}_s.svg') for name in names]),
        Col('text', 'Precision', precisions),
        Col('text', 'Recall', recalls),
        Col('text', 'F1', fmeasures),
        Col('model', 'Predict', [os.path.join('pred_mesh', f'{name}.glb') for name in names]),
        Col('model', 'GT', [os.path.join('gt_mesh', f'{name}.glb') for name in names]),
    ]

    imagetable(columns,
        out_file=os.path.join(args.exp_path, 'index.html'),
        imsize=(256, 256),
        sticky_header=True,
        sort_style='materialize',
        sortable=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', metavar="DIR", default="data",
                        help='dataset source root.')
    parser.add_argument('--exp_path', type=str, default="lightning_logs/version_X",
                        help='results path.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.exp_path, 'pred_mesh'), exist_ok=True)
    os.makedirs(os.path.join(args.exp_path, 'gt_mesh'), exist_ok=True)
    
    main()
