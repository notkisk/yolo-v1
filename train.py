"""
Main training script for YOLOv1 on Pascal VOC dataset.
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import YOLOV1
from src.dataset import VOCDataset
from src.loss import YOLOLoss

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 0 # set to 0 to avoid pickling errors on windows
PIN_MEMORY = True
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

def train_fn(train_loader, model, optimizer, loss_fn, num_classes):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        out = out.reshape(x.shape[0], 7, 7, num_classes + 10)
        
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

class ResizeAndTensoring:
    def __init__(self, size=(448, 448)):
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, img):
        img = self.resize(img)
        return self.to_tensor(img)

def main():
    NUM_CLASSES = 5
    model = YOLOV1(grid_size=7, num_boxes=2, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YOLOLoss(S=7, B=2, C=NUM_CLASSES)

    transform = Compose([ResizeAndTensoring()])

    train_dataset = VOCDataset("data/train.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR, C=NUM_CLASSES)
    test_dataset = VOCDataset("data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR, C=NUM_CLASSES)

    # limit to a small subset for quick training as requested
    train_subset = torch.utils.data.Subset(train_dataset, range(0, 100)) 
    
    train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True,)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True,)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, num_classes=NUM_CLASSES)

if __name__ == "__main__":
    main()
