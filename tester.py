# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import json
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader

from face_trainer import Trainer as FaceTrainer
from polygen.dataset import TestDataset
from polygen.data_utils import parse_splits_list, dequantize_values
from vertex_trainer import Trainer as VertexTrainer


class Tester(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)

        self.vertex_model = VertexTrainer.load_from_checkpoint(self.hparams.vertex_ckpt_path).model
        self.face_model = FaceTrainer.load_from_checkpoint(self.hparams.face_ckpt_path).model

        self.max_num_vert = self.hparams.data['max_num_vert']
        self.vertex_token = self.hparams.vertex_token
        self.face_token = self.hparams.face_token

    def test_dataloader(self):
        info_files = parse_splits_list(self.hparams.datasets_test)
        dataset = TestDataset(
            self.hparams.root, info_files, self.hparams.vertex_token, self.hparams.data)
        dataloader = DataLoader(
            dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)
        return dataloader

    def parse_vertex_sequence(self, sequence):
        valid_mask = torch.cumsum(sequence == self.vertex_token['END'], 0) == 0
        valid_seq = sequence[valid_mask]

        num_plank = len(valid_seq) // 3
        vertices = valid_seq[:num_plank * 3].reshape(-1, 3)
        vertices = torch.unique(vertices, dim=0)

        return vertices

    def parse_face_sequence(self, input):
        input = input.cpu().numpy()
        # cut off at the end of sequence, leave the EOS here
        valid = np.split(input, np.where(input == self.hparams.face_token['END'])[0]+1)[0]
        # split the sequence into group
        groups = np.split(valid, np.where(valid == self.hparams.face_token['NEW'])[0]+1)
        groups = [(group[:-1] - len(self.hparams.face_token)).tolist() for group in groups if len(group) > 3]

        return groups

    def test_step(self, batch, batch_idx):

        if batch_idx == 0:
            os.makedirs(os.path.join(self.logger.log_dir, 'results'), exist_ok=True)

        tic = time.time()

        vert_outputs = self.vertex_model(batch)

        vertices = []
        vertices_mask = []

        for vert_output in vert_outputs:
            vertex = self.parse_vertex_sequence(vert_output)

            vertex_mask = torch.ones(
                self.max_num_vert + 3, dtype=bool, device=vertex.device)
            vertex_mask[:3+len(vertex)] = False

            padding = torch.zeros(
                (self.max_num_vert - len(vertex), 3), dtype=vertex.dtype, device=vertex.device)
            vertex = torch.cat((vertex, padding), dim=0)

            vertices.append(vertex)
            vertices_mask.append(vertex_mask)

        face_inputs = {
            'vertex': torch.stack(vertices),
            'vertex_mask': torch.stack(vertices_mask)
        }

        face_outputs = self.face_model(face_inputs)

        for i in range(len(batch['name'])):

            name = batch['name'][i]

            vertex = vertices[i]
            vertex = vertex[torch.all(vertex != 0, 1)]     # removing padding vertices
            vertex = vertex.cpu().numpy()
            vertex = dequantize_values(vertex, self.hparams.data['quantization_bits'])

            pred_face = self.parse_face_sequence(face_outputs[i])

            toc =  time.time() - tic

            with open(os.path.join(self.logger.log_dir, 'results', f'{name}.json'), 'w') as f:
                json.dump({
                    'faces': pred_face,
                    'vertices': vertex.tolist(),
                    'runtime': toc
                }, f, indent=4, separators=(', ', ': '))


if __name__ == '__main__':
    cli = LightningCLI(Tester)
