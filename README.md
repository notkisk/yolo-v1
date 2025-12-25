# yolov1 implementation 

this is my implementation of yolov1 from scratch using pytorch. honestly, it's been a ride building this.

## my thoughts & critique

so, implementing yolo v1 is... interesting. the paper is legendary, but man, the loss function is kinda sketchy when you really look at it.

## getting started

dependencies are managed with `uv` (super fast, highly recommend).

1.  **get the data**  
    downloading the cppe-5 dataset (medical personal protective equipment) from huggingface:
    ```bash
    uv run prepare_data.py
    ```

2.  **train it**  
    fire up the training:
    ```bash
    uv run train.py
    ```
