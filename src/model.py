"""
yolo is basically a huge convolutional network with a few fully connected layers at the end, a classic network architecture, the unique part is the loss function and the output shape
"""

import torch
import torch.nn as nn
from .config import architecture_config

class CNNBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) # batch normalization is used to normalize the output of the previous layer
        self.leakyrelu = nn.LeakyReLU(0.1) # they use 0.1 as default, it will help to prevent the vanishing gradient problem, 0.1 is the slope of the negative part of the activation function, 
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers += [
                    CNNBLOCK(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif isinstance(x, list):
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        CNNBLOCK(
                            in_channels,
                            x[0][1],
                            kernel_size=x[0][0],
                            stride=x[0][2],
                            padding=x[0][3],
                        )
                    ]
                    layers += [
                        CNNBLOCK(
                            x[0][1],
                            x[1][1],
                            kernel_size=x[1][0],
                            stride=x[1][2],
                            padding=x[1][3],
                        )
                    ]
                    in_channels = x[1][1]
        return nn.Sequential(*layers)

    def _create_fcs(self, grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # in the original paper, they used 4096 but thats an overkill for us, we can use 496 just to make it faster
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + (B * 5))),
        )


def test_model():
    model = YOLOV1(grid_size = 7, num_boxes = 2, num_classes = 20)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)  # this will output 2 by 1470 where the 1470 is 7 * 7 so 7 by 7 grid and for each grid cell there are 30 values; 20 for class probabilities, and 2 for bounding box confidence(IoU) and 8 for bounding box coordinates of the two predicted boxes
    # so in tottal, we predict 2 bounding boxes for each grid cell, 
    # eventually, we will have to reshape the output so it become of shape (2, 7, 7, 30)
    # where 2 is the batch size, 7 is the grid size, 7 is the grid size, 30 is the number of bounding boxes

if __name__ == "__main__":
    test_model()