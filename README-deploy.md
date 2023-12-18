```bash
python3 export.py --weights runs/train/yolov7-we-v1042/weights/last.pt \
	--grid --end2end --simplify --topk-all 100 --img-size 640 640 \
	--conf-thres 0.01 --iou-thres 0.65 --grid --dynamic-batch \
	--dynamic --max-wh 640
```
Please, dont not use `--include-nms` because it is for TensorRT.