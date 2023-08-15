"""
Modified based on https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from third_party.boxes import Boxes, pairwise_iou

LARGE_COST_VALUE = 100000


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    """

    def __init__(self, threshold: float = 0.5):
        """Creates the matcher

        Params:
            threshold: matching IOU threshold
        """
        super().__init__()
        self.threshold = threshold
        assert threshold != 0, "threshold cant be 0"

    @torch.no_grad()
    def forward(self, pred_boxes, boxes):
        """ Performs the matching

        Params:
            "pred_boxes": Tensor of dim [N, 6] with the predicted box coordinates
            "boxes": Tensor of dim [M, 6] containing the target box coordinates

        Returns:
            Metric values: precision, recall, f1
        """

        num_pred = len(pred_boxes)
        num_label = len(boxes)

        # calculate IoU of all bbox pairs
        iou_matrix = pairwise_iou(boxes1=Boxes(pred_boxes), boxes2=Boxes(boxes))

        # assign large cost value to make sure pair below IoU threshold won't be matched
        cost_matrix = np.full((num_pred, num_label), LARGE_COST_VALUE)

        cost_matrix = self.assign_cost_matrix_values(cost_matrix, iou_matrix)

        indices = linear_sum_assignment(cost_matrix)

        tp = iou_matrix[torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]

        tp = torch.sum(tp >= self.threshold)

        prec = tp / num_pred if num_pred != 0 else torch.tensor(0.0)
        rec = tp / num_label if num_label != 0 else torch.tensor(0.0)
        f1 = prec * rec * 2 / (prec + rec + 1e-10)
        
        return prec, rec, f1

    def assign_cost_matrix_values(self, cost_matrix: np.ndarray, iou_matrix: torch.tensor) -> np.ndarray:
        """
        Based on IoU for each pair of bbox, assign the associated value in cost matrix
        Args:
            cost_matrix: np.ndarray, initialized 2D array with target dimensions
            iou_matrix: list of bbox pair, in each pair, iou value is stored
        Return:
            np.ndarray, cost_matrix with assigned values
        """
        iou_matrix = iou_matrix.cpu().numpy()
        cost_matrix[iou_matrix > self.threshold] = -1
        return cost_matrix


def build_matcher(threshold):
    return HungarianMatcher(threshold)
