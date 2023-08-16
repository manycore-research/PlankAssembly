# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import torch
from torchmetrics import Metric


class Criterion(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fmeasure", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prec: torch.Tensor, rec: torch.Tensor, f1: torch.Tensor):
        self.precision += prec
        self.recall += rec
        self.fmeasure += f1
        self.total += 1

    def compute(self):
        # compute final result
        return self.precision / self.total, self.recall / self.total, self.fmeasure / self.total


def build_criterion():
    return Criterion()
