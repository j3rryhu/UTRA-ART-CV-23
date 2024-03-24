from ultralytics import YOLO
import ultralytics
import gc

import torch, torch.nn as nn
from torch.nn.utils import prune

ultralytics.checks()

gc.collect()

model = YOLO("yolov8n.pt")


# Return global model sparsity: percentage of parameter pruned
def sparsity(model):
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


# Percentage of parameters to be pruned (tune this)
pruning_param = 0.05

# prune only the 2D convolutional and linear layers
for name, m in model.model.named_modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
        prune.l1_unstructured(m, name='weight', amount=pruning_param)  # prune
        prune.remove(m, 'weight')  # make permanent
print(f'Model pruned to {sparsity(model.model):.3g} global sparsity')

# save the pruned YOLOv8 model
ckpt = {

            'model': model.model,
            'train_args': {},  # save as dict
}

torch.save(ckpt, f'yolov8n_{pruning_param}.pt')

# train our final model based on our pruned model
model = YOLO(f'yolov8n_{pruning_param}.pt')
project = f'train_prune_{pruning_param}'            # named your trained model here
results = model.train(data='DETECTION-2/data.yaml', epochs=15, resume=False, device=[0], workers=4, int8=False, project=project)
model.export()

gc.collect()

# validation
metrics = model.val(data='DETECTION-2/data.yaml')
