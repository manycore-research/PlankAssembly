# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader

from polygen.data_utils import parse_splits_list
from polygen.dataset import VertexDataset
from polygen.metric import Criterion
from polygen.models import VertexModel


class Trainer(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.model = VertexModel(
            **self.hparams['model'], **self.hparams['data'], token=self.hparams.token)

        self.criterion = Criterion()

    def train_dataloader(self):
        info_files = parse_splits_list(self.hparams.datasets_train)
        dataset = VertexDataset(
            self.hparams.root, info_files, self.hparams.token, self.hparams.data, True)
        dataloader = DataLoader(
            dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        info_files = parse_splits_list(self.hparams.datasets_valid)
        dataset = VertexDataset(
            self.hparams.root, info_files, self.hparams.token, self.hparams.data)
        dataloader = DataLoader(
            dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)
        return dataloader

    def test_dataloader(self):
        info_files = parse_splits_list(self.hparams.datasets_test)
        dataset = VertexDataset(
            self.hparams.root, info_files, self.hparams.token, self.hparams.data)
        dataloader = DataLoader(
            dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)
        return dataloader

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)

        loss = torch.mean(outputs['loss'])
        accuracy = torch.mean(outputs['accuracy'])

        self.log('train/loss', loss, logger=True, batch_size=self.hparams.batch_size)
        self.log('train/accuracy', accuracy, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        return loss

    def parse_sequence(self, sequence):
        valid_mask = torch.cumsum(sequence == self.hparams.token['END'], 0) == 0
        valid_seq = sequence[valid_mask]

        num_plank = len(valid_seq) // 3
        vertices = valid_seq[:num_plank * 3].reshape(-1, 3)
        vertices = torch.unique(vertices, dim=0)
        return vertices

    def compute_metric(self, predict, label):
        
        label = self.parse_sequence(label)
        predict = self.parse_sequence(predict)

        tp = torch.sum(torch.all(
            predict.unsqueeze(0) == label.unsqueeze(1), axis=-1))

        prec = tp / len(predict) if len(predict) != 0 else torch.tensor(0.0)
        rec = tp / len(label) if len(label) != 0 else torch.tensor(0.0)

        f1 = prec * rec * 2 / (prec + rec + 1e-10)

        return prec, rec, f1

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)

        for pred, gt in zip(outputs, batch['vertex']):
            metrics = self.compute_metric(pred, gt)
            self.criterion.update(*metrics)

    def validation_epoch_end(self, batch):
        prec, rec, f1 = self.criterion.compute()

        self.log('val/precision', prec, logger=True, sync_dist=True)
        self.log('val/recall', rec, logger=True, sync_dist=True)
        self.log('val/fmeasure', f1, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        outputs = self.model(batch)

        for pred, gt in zip(outputs, batch['vertex']):
            metrics = self.compute_metric(pred, gt)
            self.criterion.update(*metrics)

    def test_epoch_end(self, batch):
        prec, rec, f1 = self.criterion.compute()

        self.log('test/precision', prec, logger=True, sync_dist=True)
        self.log('test/recall', rec, logger=True, sync_dist=True)
        self.log('test/fmeasure', f1, logger=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':
    cli = LightningCLI(Trainer)
