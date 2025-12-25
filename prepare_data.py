import os
import csv
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

# constants
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# cppe-5 class names
CLASS_NAMES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]
CLASS_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

def convert_to_yolo_box(bbox, width, height):
    # CPPE-5 on HF provides [x, y, width, height]
    x_min, y_min, w, h = bbox
    
    # Calculate normalized center x, y and normalized width, height
    x_center = (x_min + w / 2.0) / width
    y_center = (y_min + h / 2.0) / height
    w_norm = w / width
    h_norm = h / height
    
    # Clamp to [0, 1] to avoid issues with floating point precision or slightly off boxes
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))
    
    return x_center, y_center, w_norm, h_norm

def process_split(dataset, output_csv, split_name):
    print(f"processing {split_name} data...")
    csv_rows = []
    
    for i, item in enumerate(tqdm(dataset)):
        image = item["image"]
        
        # generate a unique filename
        img_id = str(i).zfill(6)
        unique_id = f"{split_name}_{img_id}"
        filename = f"{unique_id}.jpg"
        
        # save image
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        w, h = image.size
        
        objects = item["objects"]
        bboxes = objects["bbox"]
        categories = objects["category"]
        
        has_valid_objects = False
        label_filename = f"{unique_id}.txt"
        label_path = os.path.join(LABELS_DIR, label_filename)
        
        with open(label_path, "w") as f:
            for bbox, category_id in zip(bboxes, categories):
                # category_id is int
                if category_id >= len(CLASS_NAMES):
                    continue
                
                # cppe-5 bbox is [x, y, width, height] according to some sources, 
                # but 'datasets' library typically converts to [xmin, ymin, xmax, ymax].
                # Let's assume [xmin, ymin, xmax, ymax] for now.
                
                x, y, width, height = convert_to_yolo_box(bbox, w, h)
                
                f.write(f"{category_id} {x} {y} {width} {height}\n")
                has_valid_objects = True
        
        if has_valid_objects:
            image.save(os.path.join(IMAGES_DIR, filename))
            csv_rows.append([filename, label_filename])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img", "label"])
        writer.writerows(csv_rows)


def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)
        
    print("loading cppe-5 from huggingface datasets...")
    
    try:
        ds_train = load_dataset("cppe-5", split="train")
        ds_test = load_dataset("cppe-5", split="test")
    except Exception as e:
        print(f"Failed to load cppe-5: {e}")
        return

    print("preparing train set...")
    process_split(ds_train, os.path.join(DATA_DIR, "train.csv"), "trainval")
    
    print("preparing test set...")
    process_split(ds_test, os.path.join(DATA_DIR, "test.csv"), "test")
    
    print("done! data prepared.")

if __name__ == "__main__":
    main()
