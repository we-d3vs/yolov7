from collections import defaultdict
import json
import numpy as np
from pycocotools.coco import COCO
import torchvision as tv
import torch
from wedataset.session_manager_v2 import SessionManager


if __name__ == "__main__":
    
    all_images = defaultdict(set)
    for db_name in ["21buttons", "ad4", "asos", "md", "chictopia", "modcloth", "zalando", "vinted"]:
        sm = SessionManager(db_name)
        all_images[db_name] = dict(sm.session.execute("SELECT filename, id FROM images").all())
    
    # Implement error analysis of category detection for the top false positive
    #predictions = json.load(open("runs/test/yolov7-we-v1042_640_v1.1.0_test_new_training2/last_predictions.json"))
    #coco = COCO("we_data/v1.1.0_test.json")
    
    predictions = json.load(open("runs/test/yolov7-we-v1042_640_v1.1.02/last_predictions.json"))
    coco = COCO("we_data/v1.1.0.json")
    
    with open("opt_thresholds.json") as f:
        opt_thresholds = json.load(f)

    pred_category = "shoe"
    gt_category = "background"
    for category in coco.cats.values(): 
        if category["name"] == pred_category:
            pred_category_id = category["id"]
            break
    
    if gt_category != "background":
        for category in coco.cats.values(): 
            if category["name"] == gt_category:
                gt_category_id = category["id"]
                break

    tmp = []
    for x in predictions:
        image_id = x["image_id"]
        category_id = x["category_id"]
        bbox = x["bbox"]
        score = x["score"]
        if category_id == pred_category_id:
            tmp.append([image_id] + bbox + [category_id, score])
    predictions = np.array(tmp)
    predictions = predictions[np.argsort(-predictions[:, -1])]
    for pred in predictions:
        image_id, x1, y1, x2, y2, category_id, score = pred
        if score < opt_thresholds[pred_category]:
            break
        detections = np.array([[x1, y1, x2, y2, score]])
        # xywh to xyxy
        detections[:, 2] += (detections[:, 0] - 1)
        detections[:, 3] += (detections[:, 1] - 1)
        ann_ids = coco.getAnnIds([int(image_id)])
        anns = coco.loadAnns(ann_ids)
        if gt_category == "background":
            bboxes = np.array([x["bbox"] for x in anns])
        else:
            bboxes = np.array([x["bbox"] for x in anns if x["category_id"] == gt_category_id])
        fname = coco.imgs[int(image_id)]["file_name"]
        if bboxes.size == 0:
            if gt_category == "background":
                if fname.find("laion") >= 0:
                    continue
                for db_name in all_images:
                    if fname in all_images[db_name]:
                        print(f"http://we-data.wide-eyes.net:8000/image?imageid={all_images[db_name][fname]}&site={db_name} {detections[0, :4]}")
                        # print(f"http://we-data.wide-eyes.net:8000/image?imageid={all_images[db_name][fname]}&site={db_name} {score} {detections}")
                        break
            continue
        # xywh to xyxy
        bboxes[:, 2] += (bboxes[:, 0] - 1)
        bboxes[:, 3] += (bboxes[:, 1] - 1)
        ious = tv.ops.box_iou(torch.tensor(detections)[:, :4], torch.tensor(bboxes))
        ious = ious.numpy()
        iou = ious.max().item()
        if gt_category == "background":
            if iou < 0.5:
                for db_name in all_images:
                    if fname in all_images[db_name]:
                        x1, y1, x2, y2 = detections[0, :4].astype(np.int32).tolist()
                        print(f"http://we-data.wide-eyes.net:8000/bbox?site={db_name}&x1={x1}&y1={y1}&x2={x2}&y2={y2}&image_id={all_images[db_name][fname]}")
                        # print(f"http://we-data.wide-eyes.net:8000/image?imageid={all_images[db_name][fname]}&site={db_name} {score} {detections}")
                        break
                continue
        else:
            if iou > 0.5:
                for db_name in all_images:
                    if fname in all_images[db_name]:
                        x1, y1, x2, y2 = detections[0, :4].astype(np.int32).tolist()
                        print(f"http://we-data.wide-eyes.net:8000/bbox?site={db_name}&x1={x1}&y1={y1}&x2={x2}&y2={y2}&image_id={all_images[db_name][fname]}")
                        # print(f"http://we-data.wide-eyes.net:8000/image?imageid={all_images[db_name][fname]}&site={db_name} {score} {detections}")
                        break
                continue