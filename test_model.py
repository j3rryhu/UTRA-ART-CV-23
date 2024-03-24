from ultralytics import YOLO
import ultralytics

ultralytics.checks()

project = 'train_prune_0.1'  # name of model
model = YOLO(f"./{project}/train/weights/best.pt")

# generate predictions for testing data which are photos captured in obstacle course for potholes and ramps
results = model('DETECTION-2/test/images/frame_*', save=True, project=project, name='frames')

# generate predictions for the obstacle course video in 480p
results4 = model('./test-videos/20230605_163120-480.mov', stream=True, save=True, verbose=True, conf=0.5, project=project, name='480p')
count = 0
for r in results4:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs
    count += 1

# generate predictions for the obstacle course video in 480p
results5 = model('./test-videos/20230605_163120-720.mov', stream=True, save=True, verbose=True, conf=0.5, project=project, name='720p')
count = 0
for r in results5:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs
    count += 1

exit(0)
