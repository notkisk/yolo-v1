import torch
import torch.nn as nn
from .utils import calculate_intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self, S, B, C):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S # grid size (e.g. 7x7)
        self.B = B # number of boxes per cell
        self.C = C # number of classes
        self.lambda_noobj = 0.5 # we care less about cells with no objects
        self.lambda_coord = 5 # we care a lot about getting the box position right

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # predictions are (batch, S, S, C + B*5)
        
        # c is the number of classes we're trying to predict
        C = self.C
        
        # yolo predicts B boxes, but only one can be "responsible" for an object.
        # we check which one has the best iou with the actual ground truth.
        iou_b1 = calculate_intersection_over_union(predictions[..., C+1:C+5].reshape(-1, 4), target[..., C+1:C+5].reshape(-1, 4))
        iou_b2 = calculate_intersection_over_union(predictions[..., C+6:C+10].reshape(-1, 4), target[..., C+1:C+5].reshape(-1, 4))
        
        iou_b1 = iou_b1.reshape(predictions.shape[0], self.S, self.S)
        iou_b2 = iou_b2.reshape(predictions.shape[0], self.S, self.S)
        
        # glue them together to find the winner
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # (2, N, S, S)
        
        # bestbox tells us if box 0 or box 1 won (0 or 1 index)
        iou_maxes, bestbox = torch.max(ious, dim=0) # (N, S, S)
        
        # does an object actually exist in this cell? (val is 1 if yes, 0 if no)
        exists_box = target[..., C].unsqueeze(3) # (N, S, S, 1)

        # --- box coordinate loss ---
        # we only penalize the box that was "responsible" (the one with best iou)
        box_predictions = exists_box * (
            (
                bestbox.unsqueeze(3) * predictions[..., C+6:C+10]
                + (1 - bestbox.unsqueeze(3)) * predictions[..., C+1:C+5]
            )
        )

        box_targets = exists_box * target[..., C+1:C+5]

        # yolo says take the sqrt of width and height so small errors in small boxes 
        # count more than small errors in big boxes. clever trick.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # calculate how much we messed up the x,y,w,h
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        
        # --- object confidence loss ---
        # we only care about the confidence of the box that won
        pred_box = (
            bestbox.unsqueeze(3) * predictions[..., C+5:C+6] + (1 - bestbox.unsqueeze(3)) * predictions[..., C:C+1]
        )

        # loss for cells where an object actually is
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., C:C+1]),
        )

        # --- no object confidence loss ---
        # if there's no object, both boxes should have 0 confidence.
        # we tone this down with lambda_noobj so it doesn't overwhelm the other losses.
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., C:C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., C:C+1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., C+5:C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., C:C+1], start_dim=1)
        )

        # --- class probability loss ---
        # if an object is there, did we guess the right class?
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :C], end_dim=-2),
            torch.flatten(exists_box * target[..., :C], end_dim=-2),
        )

        # sum it all up with our weights. coordinates are important!
        loss = (
            self.lambda_coord * box_loss 
            + object_loss 
            + self.lambda_noobj * no_object_loss 
            + class_loss 
        )

        return loss