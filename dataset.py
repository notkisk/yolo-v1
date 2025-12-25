import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=5, transform=None
    ):
        # we load a csv that tells us which image matches which label file
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S # grid size
        self.B = B # num boxes
        self.C = C # num classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # get the label file for this image
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                # each line is: class_id x_center y_center width height (all normalized 0-1)
                class_label, x, y, width, height = [
                    float(x) for x in line.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        # load the image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        # apply data augmentation if we have any
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # yolo divides the image into an SxS grid. 
        # we need to convert our list of boxes into a matrix that matches this grid.
        # the matrix stores [class_probs..., confidence, x, y, w, h] for each cell.
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # figure out which cell the center of the box falls into
            i, j = int(self.S * y), int(self.S * x)
            i = min(i, self.S - 1)
            j = min(j, self.S - 1)
            # x_cell and y_cell are the relative coordinates within that specific cell (0-1)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            # width and height are scaled relative to the whole image, but we scale them by S 
            # so they are relative to the cell size for the model to predict easier
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # yolo only allows one object to be responsible per cell.
            # if we already put an object in this cell, we just skip the rest.
            if label_matrix[i, j, self.C] == 0:
                # set confidence to 1 because there's an object here
                label_matrix[i, j, self.C] = 1

                # put the box coordinates in the right spot
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates

                # set the one-hot encoding for the class
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
