import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, S, B, C):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5 # according to the paper, the authors used 5 for lambda_coord and 0.5 for lambda_noobj
        # based on my understanding of this weird loss function, the lambda_coord amplifies the coordinate loss so it doesn't get drowned out by the confidence loss, because that loss term is present for every single anchor box even if there is no object it will be present and equal to something like (p -0)²
        # the lambda_noobj tone down the no object loss so its less important than the localization loss, at the end we trying to get the best localization possible while also trying to predict the correct bboxes(thats what confidence loss does, it uses the iou to calculate the loss)

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # predictions is of shape (N, S, S, 30) where N is the batch size, S is the grid size, 30 is the number of bounding boxes
        # target is of shape (N, S, S, 30)
        ...