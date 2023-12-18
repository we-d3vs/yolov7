# 
from nni.compression.speedup import ModelSpeedup
from nni.compression.pruning import L1NormPruner

import torch
from models.experimental import attempt_load

# Load the original model and the pruned model
model = attempt_load("runs/train/yolov7-we-v1042/weights/last.pt", map_location="cuda")  # load FP32 model
model.eval()

config_list = [{'op_types': ['Conv2d'], 'sparsity': 0.8}]
pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()


pruner.unwrap_model()
model = ModelSpeedup(model, dummy_input=(torch.rand(10, 3, 640, 640).to("cuda"), None), masks_or_file=masks).speedup_model()
print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
# print('Pruned model without finetuning acc: ', evaluate(model, test_loader), '%')