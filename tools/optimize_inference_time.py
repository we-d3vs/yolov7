import torch
from models.experimental import attempt_load, End2End

import os

tensorrt_path = "~/VENV_PY10/lib/python3.10/site-packages/tensorrt"  # Change this path according to your TensorRT location

if os.path.exists(tensorrt_path):
    os.environ['LD_LIBRARY_PATH'] += f":{tensorrt_path}"
else:
    print("Unable to find TensorRT path. ONNXRuntime won't use TensorrtExecutionProvider.")

device = torch.device("cuda")

model = attempt_load("runs/train/yolov7-we-v1042/weights/last.pt",
                     map_location="cpu")  # load FP32 model

model.to(device)
model.eval()

test_data = torch.randn(1, 3, 640, 640)


with torch.no_grad():
    out = model(test_data.to(device))  # run once

"""
The original YOLOv8 model return as output a tuple
where the first element is a tensor and the second is a list of tensors.
Speedster currently supports only models that return only tensors,
so we need to create a wrapper to overcome this issue:
"""

class YOLOWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model
    
    def forward(self, x, *args, **kwargs):
        res = self.model(x)
        return res[0], res[1][0], res[1][1], res[1][2]


model_wrapper = YOLOWrapper(model)
with torch.no_grad():
    out = model_wrapper(test_data.to(device))  # run once


from speedster import optimize_model, save_model, load_model

input_data = [((torch.randn(1, 3, 640, 640), ), torch.tensor([0])) for i in range(100)]

# Run Speedster optimization
optimized_model = optimize_model(
  model_wrapper, input_data=input_data,
  metric_drop_ths=0.01,
  store_latencies=True
)


class OptimizedYOLO(torch.nn.Module):
    def __init__(self, optimized_model):
        super().__init__()
        self.model = optimized_model
    
    def forward(self, x, *args, **kwargs):
        res = self.model(x)
        return res[0], list(res[1:])
    
optimized_wrapper = OptimizedYOLO(optimized_model)

optimized_wrapper(test_data.cuda())


save_model(optimized_model, "yolov7_optimized.pth")
