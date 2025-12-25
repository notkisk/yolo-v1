# yolov1 implementation 

this is my implementation of yolov1 from scratch using pytorch.

## my thoughts & critique

implementing yolo v1 is honestly a really interesting experience. the paper itself is legendary, but once you actually try to implement it, you start to see how flawed the loss function really is. using sum squared error for everything including classification and confidence just feels wrong, especially when you understand that these are fundamentally classification problems where bce makes much more sense. while working through the implementation and debugging the behavior of the model, it became clear that a lot of the loss design relies on hardcoded scaling factors like lambda_coord and lambda_noobj just to keep training from collapsing. those scalers feel more like manual patches than something that naturally falls out of the model design. going through this process really helped me understand why the loss function ended up being one of the most heavily redesigned parts in later yolo versions

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
