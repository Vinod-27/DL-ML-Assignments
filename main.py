import torch

# Model
model = torch.hub.load('D:\DL Assignment\yolov5','custom', r'D:\DL Assignment\models\best.pt',source='local')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = r'D:\DL Assignment\test_images\15.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()
results.pandas()
results.save()
# Model
