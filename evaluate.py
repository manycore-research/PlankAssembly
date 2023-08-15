# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from plankassembly.data import dequantize_values
from plankassembly.metric import build_criterion
from third_party.matcher import build_matcher


def main(args):
    filenames = os.listdir(os.path.join(args.exp_path, 'pred_jsons'))

    criterion = build_criterion()

    metrics = dict()

    matcher = build_matcher(args.threshold)

    for filename in tqdm(filenames):
        name = filename.split('.')[0]
        
        with open(os.path.join(args.exp_path, 'pred_jsons', filename)) as f:
            pred_data = json.load(f)

        with open(os.path.join(args.data_path, 'infos', filename), 'r') as f:
            gt_data = json.load(f)

        pred = np.array(pred_data['prediction'])
        pred = dequantize_values(pred, args.num_bits)

        gt = np.array(gt_data['coords'])

        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)

        prec, recal, f1 = matcher(pred[1:], gt[1:])
        criterion.update(prec, recal, f1)

        metrics[name] = {
            'precision': prec.numpy().tolist(),
            'recall': recal.numpy().tolist(),
            'fmeasure': f1.numpy().tolist()
        }

    json.dump(metrics, open(os.path.join(args.exp_path, 'metrics.json'), 'w'))

    prec, recal, fscore = criterion.compute()

    print('%10s %0.3f' % ('prec', prec * 100))
    print('%10s %0.3f' % ('rec', recal * 100))
    print('%10s %0.3f' % ('f1', fscore * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', metavar="DIR", default="data/data/complete",
                        help='dataset source root.')
    parser.add_argument('--exp_path', type=str, default="lightning_logs/test_line_noise_0",
                        help='log path.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="threshold")
    parser.add_argument("--num_bits", type=int, default=9,
                        help="number of bits")
    args = parser.parse_args()

    main(args)
