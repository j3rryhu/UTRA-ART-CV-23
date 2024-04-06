from ultralytics import YOLO
import ultralytics
import gc

ultralytics.checks()

gc.collect()

model = YOLO("yolov8n.pt")
project = 'train_v2dataset_int'         # name your trained model here

# Train a model without quantization or pruning
results = model.train(data='DETECTION-2/data.yaml', epochs=15, resume=False, device=[0], workers=4, project=project)

# Train a model with int8 quantization
# results = model.train(data='DETECTION-2/data.yaml', epochs=15, resume=False, device=[0], workers=4, int8=True, project=project)

model.export()

gc.collect()

# validation
metrics = model.val(data='DETECTION-2/data.yaml')
