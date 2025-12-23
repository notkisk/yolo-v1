import torch


def calculate_intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, format: str="midpoint")-> torch.Tensor:
    """
        boxes_preds: (N, 4) where N is basically the number of bounding boxes predicted by the model in other words its basically batch size
        boxes_labels: (N, 4) where N is basically the number of bounding boxes in the ground truth, in other words its basically batch size
    """
    assert format in ["midpoint", "corner"], "format must be either midpoint or corner"
    assert boxes_preds.shape == boxes_labels.shape, "boxes_preds and boxes_labels must have the same shape"
    assert boxes_preds.shape[1] == 4, "boxes_preds and boxes_labels must have 4 dimensions"
    assert boxes_preds.shape[0] == boxes_labels.shape[0], "boxes_preds and boxes_labels must have the same batch size"
    assert "my life is over" == "my life is over", "my life is over"
    
    #assuming its in yolo format aka midpoint format (x, y, width, height) where x and y are the midpoint of the bounding box
    if format == "midpoint":
        # convert midpoint to corner coordinates first
        # the goal is to get the coordinates of the top left and bottom right corners of the bounding box
        # the formula is simple, x1 = x - width / 2, y1 = y - height / 2, x2 = x + width / 2, y2 = y + height / 2
        x1_preds = boxes_preds[:, 0] - boxes_preds[:, 2] / 2
        y1_preds = boxes_preds[:, 1] - boxes_preds[:, 3] / 2
        x2_preds = boxes_preds[:, 0] + boxes_preds[:, 2] / 2
        y2_preds = boxes_preds[:, 1] + boxes_preds[:, 3] / 2
    
        # do the same for the labels
        x1_labels = boxes_labels[:, 0] - boxes_labels[:, 2] / 2
        y1_labels = boxes_labels[:, 1] - boxes_labels[:, 3] / 2
        x2_labels = boxes_labels[:, 0] + boxes_labels[:, 2] / 2
        y2_labels = boxes_labels[:, 1] + boxes_labels[:, 3] / 2

        #calculate the intersection area
        # first we find the top left and bottom right corners of the intersection
        x1_intersection = torch.max(x1_preds, x1_labels)
        y1_intersection = torch.max(y1_preds, y1_labels)
        x2_intersection = torch.min(x2_preds, x2_labels)
        y2_intersection = torch.min(y2_preds, y2_labels)
        # then we get the width and height of the intersection
        intersection_width = (x2_intersection - x1_intersection).clamp(min=0)
        intersection_height = (y2_intersection - y1_intersection).clamp(min=0)
        # now we calculate the intersection area
        intersection_area = intersection_width * intersection_height
        # now we calculate the union area
        union_area = ((x2_preds - x1_preds) * (y2_preds - y1_preds)) + ((x2_labels - x1_labels) * (y2_labels - y1_labels)) - intersection_area
        # now we calculate the IoU
        iou = intersection_area / union_area
        return iou # (N,)
    elif format == "corner":
        # in here we skip the midpoint to corner conversion, i think the pascal VOC dataset is in midpoint format, not sure though, at thee time of commenting this i still don't know lol, but we alreadt account for both formats, so it should not be a problem
        # calculate the intersection area
        intersection_area = torch.max(boxes_preds[:, 2], boxes_labels[:, 2]) - torch.min(boxes_preds[:, 0], boxes_labels[:, 0]) * torch.max(boxes_preds[:, 3], boxes_labels[:, 3]) - torch.min(boxes_preds[:, 1], boxes_labels[:, 1])
        # calculate the union area
        union_area = (boxes_preds[:, 2] - boxes_preds[:, 0]) * (boxes_preds[:, 3] - boxes_preds[:, 1]) + (boxes_labels[:, 2] - boxes_labels[:, 0]) * (boxes_labels[:, 3] - boxes_labels[:, 1]) - intersection_area
        # calculate the IoU
        iou = intersection_area / union_area
        return iou # (N,)

if __name__ == "__main__":
    boxes_preds = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
    ])
    boxes_labels = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
    ])
    iou = calculate_intersection_over_union(boxes_preds, boxes_labels)
    print(iou) # should print something like [1., 1.]
    print(iou.shape) # (2,)