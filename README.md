# UTRA-ART-CV-23
# YOLO models for Potholes and Ramp

## Dataset
Here is the link for the dataset: https://app.roboflow.com/potholetestingdata/detection-y7j2b/2. 
It consists of photos with potholes and ramp captured in the obstacle course.

## Models Performance
YOLOv8n is used as base model.

### RTX6000
- Test Images: Speed: 2.0ms preprocess, 8.7ms inference, 1.0ms postprocess per image at shape (1, 3, 640, 640)
- 480p: Speed: 1.0ms preprocess, 8.3ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
- 720p: Speed: 1.6ms preprocess, 8.8ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)



- Validation accuracy:

Class | Images | Instances | Box(P | R | mAP50 | mAP50-95)
--- | --- | --- | --- | --- | --- | ---
all | 166 | 204 | 0.994 | 0.976 | 0.989 | 0.885
Potholes | 166 | 123 | 1 | 0.964 | 0.993 | 0.875
ramp | 166 | 81 | 0.987 | 0.988 | 0.986 | 0.895

### RTX6000 (Quantitization)
- Test Images: Speed: 2.0ms preprocess, 8.7ms inference, 1.0ms postprocess per image at shape (1, 3, 640, 640)
- 480p: Speed: 1.0ms preprocess, 8.3ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
- 720p: Speed: 1.6ms preprocess, 8.9ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)



- Validation accuracy:

Class | Images | Instances | Box(P | R     | mAP50 | mAP50-95)
--- | --- |-----------|-------|-------|-------| --- 
all | 166 | 204       | 0.994 | 0.976 | 0.989 | 0.885
Potholes | 166 | 123       | 1     | 0.964 | 0.993 | 0.875
ramp | 166 | 81        | 0.987 | 0.988 | 0.986 | 0.895



### RTX6000 (Pruning 0.05)
- Model summary (fused): 168 layers, 3006038 parameters, 0 gradients, 8.1 GFLOPs
- Test Images: Speed: 2.1ms preprocess, 8.8ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 640)
- 480p: Speed: 1.0ms preprocess, 8.4ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
- 720p: Speed: 1.7ms preprocess, 9.0ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)



- Validation accuracy:

Class | Images | Instances | Box(P | R     | mAP50 | mAP50-95)
--- | --- |-----------|-------|-------|-------| ---
all | 166 | 204       | 0.979 | 0.988 | 0.99 | 0.881
Potholes | 166 | 123       | 0.984 | 0.989 | 0.995 | 0.867
ramp | 166 | 81        | 0.974 | 0.988 | 0.984 | 0.896



### RTX6000 (Pruning 0.1)
Model summary (fused): 168 layers, 3006038 parameters, 0 gradients, 8.1 GFLOPs (accracy dropped quite a lot)
- Test Images: Speed: 2.2ms preprocess, 8.9ms inference, 1.2ms postprocess per image at shape (1, 3, 640, 640)
- 480p: Speed: 0.9ms preprocess, 8.5ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
- 720p: Speed: 1.7ms preprocess, 9.1ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)



- Validation accuracy:

Class | Images | Instances | Box(P | R     | mAP50 | mAP50-95)
--- | --- |-----------|-------|-------|-------| ---
all | 166 | 204       | 0.688 | 0.749 | 0.788 | 0.604
Potholes | 166 | 123       | 0.622 | 0.732 | 0.711 | 0.519
ramp | 166 | 81        | 0.753 | 0.765 | 0.865 | 0.69


### Conclusion
It is hard to further optimize the existing model because YOLOv8n is already used as base model, which is the most compact YOLOv8 model. 
The extra inference time introduced in the optimized models is suspected to be overhead as the size of the model is not greatly reduced.
