# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
"""
Modified based on https://github.com/magicleap/Atlas/blob/master/train.py
* We replace the head in atlas with a Transformer decoder takes the flattened features as input and outputs the shape program.
"""
import itertools
import json
import os

import pytorch_lightning as pl
import torch
from detectron2.config import CfgNode
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader

import atlas.transforms as transforms
from atlas.models.backbone2d import build_backbone2d
from atlas.models.backbone3d import build_backbone3d
from atlas.utils import backproject
from plankassembly.data import LineDataset, collate_fn, parse_splits_list
from plankassembly.metric import build_criterion
from plankassembly.model import build_model
from third_party.matcher import build_matcher


class Trainer(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        
        cfg = CfgNode(hparams)
        self.cfg = cfg

        # networks
        self.backbone2d, self.backbone2d_stride = build_backbone2d(cfg)
        self.backbone3d = build_backbone3d(cfg)
        self.heads3d = build_model(cfg)

        self.matcher = build_matcher(cfg.THRESHOLD)

        self.criterion = build_criterion()

        self.voxel_dim = [cfg.MODEL.VOXEL_DIM, ] * 3
        self.voxel_size = 2 / cfg.MODEL.VOXEL_DIM
        self.origin = torch.tensor([-1,-1,-1]).view(1,3)

        self.batch_backbone2d_time = cfg.MODEL.BATCH_BACKBONE2D_TIME

        self.initialize_volume()

    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """

        self.volume = 0

    def inference1(self, projection, image=None, feature=None):
        """ Backprojects image features into 3D and accumulates them.
        This is the first half of the network which is run on every frame.
        Only pass one of image or feature. If image is passed 2D features
        are extracted from the image using self.backbone2d. When features
        are extracted external to this function pass features (used when 
        passing multiple frames through the backbone2d simultaniously
        to share BatchNorm stats).
        Args:
            projection: bx3x4 projection matrix
            image: bx3xhxw RGB image
            feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)
        Feature volume is accumulated into self.volume and self.valid
        """

        assert ((image is not None and feature is None) or 
                (image is None and feature is not None))

        if feature is None:
            # image = self.normalizer(image)
            feature = self.backbone2d(image)

        # backbone2d reduces the size of the images so we 
        # change intrinsics to reflect this
        projection = projection.clone()
        projection[:,:2,:] = projection[:,:2,:] / self.backbone2d_stride

        volume = backproject(self.voxel_dim, self.voxel_size, self.origin,
                             projection, feature)

        self.volume = self.volume + volume

    def inference2(self, targets=None):
        """ Refines accumulated features and regresses output TSDF.
        This is the second half of the network. It should be run once after
        all frames have been accumulated. It may also be run more fequently
        to visualize incremental progress.
        Args:
            targets: used to compare network output to ground truth
        Returns:
            tuple of dicts ({outputs}, {losses})
                if targets is None, losses is empty
        """

        volume = self.volume

        # remove nans (where self.valid==0)
        # volume = volume.transpose(0,1)
        # volume[:,self.valid.squeeze(1)==0]=0
        # volume = volume.transpose(0,1)

        x = self.backbone3d(volume)

        return self.heads3d(x, targets)

    def forward(self, batch):
        """ Wraps inference1() and inference2() into a single call.
        Args:
            batch: a dict from the dataloader
        Returns:
            see self.inference2
        """

        self.initialize_volume()

        image = batch['image']
        projection = batch['projection']

        # get targets if they are in the batch
        targets3d = {key: value for key, value in batch.items() if key.startswith('output')}
        targets3d = targets3d if targets3d else None

        # transpose batch and time so we can accumulate sequentially
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)

        if (not self.batch_backbone2d_time) or (not self.training):
            # run images through 2d cnn sequentially and backproject and accumulate
            for image, projection in zip(images, projections):
                self.inference1(projection, image=image)

        else:
            # run all images through 2d cnn together to share batchnorm stats
            image = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:])
            features = self.backbone2d(image)

            # reshape back
            features = features.view(images.shape[0], images.shape[1], *features.shape[1:])

            for projection, feature in zip(projections, features):
                self.inference1(projection, feature=feature)

        # run 3d cnn
        outputs = self.inference2(targets3d)

        return outputs

    def get_transform(self):
        """ Gets a transform to preprocess the input data"""

        transform = [
            transforms.ToTensor(),
            transforms.IntrinsicsPoseToProjection(),
        ]

        return transforms.Compose(transform)

    def train_dataloader(self):
        transform = self.get_transform()
        info_files = parse_splits_list(self.cfg.DATASETS_TRAIN)
        dataset = LineDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, transform, self.cfg.DATA, augmentation=True)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS, collate_fn=collate_fn,
            shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        transform = self.get_transform()
        info_files = parse_splits_list(self.cfg.DATASETS_VALID)
        dataset = LineDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, transform, self.cfg.DATA)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS, collate_fn=collate_fn)
        return dataloader

    def test_dataloader(self):
        transform = self.get_transform()
        info_files = parse_splits_list(self.cfg.DATASETS_TEST)
        dataset = LineDataset(
            self.cfg.ROOT, info_files, self.cfg.TOKEN, transform, self.cfg.DATA)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS, collate_fn=collate_fn)
        return dataloader

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        loss = torch.mean(outputs['loss'])
        accuracy = torch.mean(outputs['accuracy'])

        self.log('train/loss', loss, logger=True, batch_size=self.cfg.BATCH_SIZE)
        self.log('train/accuracy', accuracy, prog_bar=True, logger=True, batch_size=self.cfg.BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        for pred, gt in zip(outputs['predicts'], outputs['groundtruths']):

            # filter invalid prediction
            valid_mask = torch.all(torch.abs(pred[1:, 3:] - pred[1:, :3]) != 0, dim=1)

            prec, rec, f1 = self.matcher(pred[1:][valid_mask], gt[1:])
            self.criterion.update(prec, rec, f1)

    def validation_epoch_end(self, batch):
        prec, rec, f1 = self.criterion.compute()

        self.log('val/precision', prec, logger=True, sync_dist=True)
        self.log('val/recall', rec, logger=True, sync_dist=True)
        self.log('val/fmeasure', f1, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        if batch_idx == 0:
            os.makedirs(os.path.join(self.logger.log_dir, 'pred_jsons'), exist_ok=True)
        
        for name, pred, gt in zip(batch['name'], outputs['predicts'], outputs['groundtruths']):
            # filter invalid prediction
            valid_mask = torch.all(torch.abs(pred[1:, 3:] - pred[1:, :3]) != 0, dim=1)
            valid_pred = torch.concat((pred[:1], pred[1:][valid_mask]))

            prec, rec, f1 = self.matcher(valid_pred[1:], gt[1:])
            self.criterion.update(prec, rec, f1)

            pred = pred.cpu().numpy().reshape(-1, 6).tolist()
            gt = gt.cpu().numpy().reshape(-1, 6).tolist()

            with open(os.path.join(self.logger.log_dir, 'pred_jsons', f'{name}.json'), 'w') as f:
                json.dump({
                    "prediction": pred,
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
        
        # allow for different learning rates between pretrained layers 
        # (resnet backbone) and new layers (everything else).
        params_heads3d = self.heads3d.parameters()
        modules = [self.backbone2d, self.backbone3d]
        params_backone = itertools.chain(*(params.parameters() for params in modules))

        optimizer = torch.optim.Adam([
                {'params': params_heads3d, 'lr': self.cfg.LR_HEAD},
                {'params': params_backone, 'lr': self.cfg.LR_BACKBONE}])

        return optimizer


if __name__ == '__main__':
    cli = LightningCLI(Trainer)
