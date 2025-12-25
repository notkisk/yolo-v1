# yolov1 implementation 

this is my implementation of yolov1 from scratch using pytorch.

## my thoughts & critique

so, implementing yolo v1 is... interesting. the paper is legendary, but the loss function is fundamentally flawed when you really look at it. using sum-squared error for everything—even classification and confidence—feels wrong;bce would have been much more appropriate in some loss terms. the reliance on hardcoded scalers like `lambda_coord` and `lambda_noobj` to balance the loss terms feels artificial, rather than a natural consequence of the model's architecture. thats why the loss function was one of the most drastically overhauled components in subsequent yolo versions

## getting started

dependencies are managed with `uv` (super fast, highly recommend).

1.  **get the data**  
    downloading the cppe-5 dataset (medical personal protective equipment) from huggingface:
    ```bash
    uv run scripts/prepare_data.py
    ```

2.  **train it**  
    ```bash
    uv run train.py
    ```
