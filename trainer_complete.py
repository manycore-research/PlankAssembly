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
from plankassembly.datasets import LineDataset
from plankassembly.metric import build_criterion
from plankassembly.models import build_model
from third_party.matcher import build_matcher


class Trainer(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        
        cfg = CfgNode(hparams)
        self.cfg = cfg

        self.model = build_model(cfg)

        self.matcher = build_matcher(cfg.THRESHOLD)

        self.criterion = build_criterion()

    def train_dataloader(self):
        info_files = parse_splits_list(self.cfg.DATASETS_TRAIN)
        dataset = LineDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, self.cfg.DATA, True)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS,
            shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        info_files = parse_splits_list(self.cfg.DATASETS_VALID)
        dataset = LineDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, self.cfg.DATA)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS)
        return dataloader

    def test_dataloader(self):
        info_files = parse_splits_list(self.cfg.DATASETS_TEST)
        dataset = LineDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, self.cfg.DATA)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS)
        return dataloader

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)

        loss = torch.mean(outputs['loss'])
        accuracy = torch.mean(outputs['accuracy'])

        self.log('train/loss', loss, logger=True, batch_size=self.cfg.BATCH_SIZE)
        self.log('train/accuracy', accuracy, prog_bar=True, logger=True, batch_size=self.cfg.BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)

        for pred, gt in zip(outputs['predicts'], outputs['groundtruths']):

            # filter invalid prediction
            valid_mask = torch.all(torch.abs(pred[1:, 3:] - pred[1:, :3]) != 0, dim=1)

            prec, rec, f1, _, _ = self.matcher(pred[1:][valid_mask], gt[1:])
            self.criterion.update(prec, rec, f1)

    def validation_epoch_end(self, batch):
        prec, rec, f1 = self.criterion.compute()

        self.log('val/precision', prec, logger=True, sync_dist=True)
        self.log('val/recall', rec, logger=True, sync_dist=True)
        self.log('val/fmeasure', f1, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        outputs = self.model(batch)

        if batch_idx == 0:
            os.makedirs(os.path.join(self.logger.log_dir, 'pred_jsons'), exist_ok=True)

        for name, pred, gt, atta in zip(batch['name'], outputs['predicts'], outputs['groundtruths'], outputs['attach']):

            # filter invalid prediction
            valid_mask = torch.all(torch.abs(pred[1:, 3:] - pred[1:, :3]) != 0, dim=1)
            valid_pred = torch.concat((pred[:1], pred[1:][valid_mask]))

            prec, rec, f1 = self.matcher(valid_pred[1:], gt[1:])
            self.criterion.update(prec, rec, f1)

            atta = atta[:len(valid_pred.flatten())].cpu().numpy().reshape(-1, 6).tolist()
            pred = valid_pred.cpu().numpy().reshape(-1, 6).tolist()
            gt = gt.cpu().numpy().reshape(-1, 6).tolist()

            with open(os.path.join(self.logger.log_dir, 'pred_jsons', f'{name}.json'), 'w') as f:
                json.dump({
                    "prediction": pred,
                    "attach": atta,
                    "groundtruth": gt,
                    "precision": prec.item(),
                    "recall": rec.item(),
                    "fmeasure": f1.item(),
                }, f, indent=4, separators=(', ', ': '))

    def test_epoch_end(self, batch):
        prec, rec, f1 = self.criterion.compute()

        self.log('test/precision', prec, logger=True, sync_dist=True)
        self.log('test/recall', rec, logger=True, sync_dist=True)
        self.log('test/fmeasure', f1, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.LR)
        return {'optimizer': optimizer}


if __name__ == '__main__':
    cli = LightningCLI(Trainer)
