# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader

from dataset.data_utils import parse_splits_list
from plankassembly.datasets import LineDataset
from trainer_complete import Trainer


class VisibleTrainer(Trainer):

    def __init__(self, hparams):
        super().__init__(hparams)

    def train_dataloader(self):
        info_files = parse_splits_list(self.cfg.DATASETS_TRAIN)
        dataset = LineDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, cfg=self.cfg.DATA)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS,
            shuffle=True, drop_last=True)
        return dataloader


if __name__ == '__main__':
    cli = LightningCLI(VisibleTrainer)
