# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
from detectron2.config import CfgNode
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader

from dataset.data_utils import parse_splits_list
from plankassembly.datasets import SidefaceDataset
from plankassembly.metric import build_criterion
from plankassembly.models import build_model
from third_party.matcher import build_matcher
from trainer_complete import Trainer


class SidefaceTrainer(Trainer):

    def __init__(self, hparams):
        super().__init__(hparams)

    def train_dataloader(self):
        info_files = parse_splits_list(self.cfg.DATASETS_TRAIN)
        dataset = SidefaceDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, self.cfg.DATA, True)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS,
            shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        info_files = parse_splits_list(self.cfg.DATASETS_VALID)
        dataset = SidefaceDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, self.cfg.DATA)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS)
        return dataloader

    def test_dataloader(self):
        info_files = parse_splits_list(self.cfg.DATASETS_TEST)
        dataset = SidefaceDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, self.cfg.DATA)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS)
        return dataloader

    def test_step(self, batch, batch_idx):
        outputs = self.model(batch)

        if batch_idx == 0:
            os.makedirs(os.path.join(self.logger.log_dir, 'pred_jsons'), exist_ok=True)

        for name, mask, pred, gt in zip(batch['name'], batch['input_mask'], outputs['predicts'], outputs['groundtruths']):

            if torch.all(mask[1:]):
                # no detected sidefaces
                pred = []
                gt = gt.cpu().numpy().reshape(-1, 6).tolist()
                prec, rec, f1 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            else:
                # filter invalid prediction
                valid_mask = torch.all(torch.abs(pred[1:, 3:] - pred[1:, :3]) != 0, dim=1)
                valid_pred = torch.concat((pred[:1], pred[1:][valid_mask]))

                prec, rec, f1, _, _ = self.matcher(valid_pred[1:], gt[1:])
                self.criterion.update(prec, rec, f1)

                pred = valid_pred.cpu().numpy().reshape(-1, 6).tolist()
                gt = gt.cpu().numpy().reshape(-1, 6).tolist()

            with open(os.path.join(self.logger.log_dir, 'pred_jsons', f'{name}.json'), 'w') as f:
                json.dump({
                    "prediction": pred,
                    "groundtruth": gt,
                    "precision": prec.item(),
                    "recall": rec.item(),
                    "fmeasure": f1.item(),
                }, f, indent=4, separators=(', ', ': '))


if __name__ == '__main__':
    cli = LightningCLI(SidefaceTrainer)
