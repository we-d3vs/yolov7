"""Use this command to run the streamlit for comparing detector demos
```
streamlit run demo_st.py \
   -- --onnx {snapshot1.onnx} {snapshot2.onnx}
```
"""
from typing import List
import argparse
from collections import defaultdict
import json
import numpy as np
import PIL
from PIL import ImageDraw
import cv2
import onnxruntime as ort
import torch
import torchvision as tv
import streamlit as st
from wedataset.sqlite_model_attr11_v2 import Image
from wedataset.session_manager_v2 import SessionManager


st.set_page_config(layout="wide", page_title=f'Detector comparison')


@st.cache(allow_output_mutation=True)
def load_detector(onnx_fname):
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_fname, providers=providers)
    return session


def draw_rectangle(pil_image, box: np.ndarray, color_hex: str):
    """Draw a rectangle on a PIL image.
    
    args:
        pil_image: PIL image
        box: [xmin, ymin, xmax, ymax]
        color_hex: color in hex format
        
    returns:
        PIL image
    """
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle(box.tolist(), outline=color_hex, width=3)
    draw.rectangle(box.tolist(), outline="black", width=1)
    return pil_image


CLASSES = [
    "top",
    "vest",
    "one-piece",
    "bottom",
    "headwear",
    "underwear",
    "swimwear",
    "jewellery",
    "accessory",
    "shoe",
    "bag",
    "knitwear|sweatshirt",
    "shorts",
    "skirt",
    "jacket",
    "coat"
]


np.random.seed(0)
COLORS = {name : [np.random.randint(0, 255) for _ in range(3)] for name in CLASSES}


IMAGE_SIZE = 400


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def apply_detectors(image, image_filename, models):

    st.sidebar.image(image, width=IMAGE_SIZE//2)
    ## Title.
    st.write('# Dakota Category Object Detection')
    cols = st.columns(len(models) + 1)

    img = image.copy()
    img, ratio, dwdh = letterbox(img, auto=False)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img)

    im = img.astype(np.float32)
    im /= 255

    for col, (model_name, model) in enumerate(models.items()):
        result = model.run(["output"], {"images": im})[0]
        im2show = image.copy()
        result2 = result.copy()
        for i,(batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(result):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 2)
            if score < confidence_threshold:
                continue
            name = CLASSES[cls_id]
            color = COLORS[name]
            name += ' ' + str(score)
            thickness = max(2, int(2 * max(im2show.shape[:2]) / 640))
            scale = 0.75 * max(im2show.shape[:2]) / 640
            print(box)
            cv2.rectangle(im2show, box[:2], box[2:], color, thickness)
            cv2.putText(im2show, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness=thickness)
        # apply nms to all the boxes
        pred_boxes = ((result2[:, 1:5] - np.array(dwdh*2)) / ratio).astype(np.float32)
        pred_scores = result2[:, -1]
        indexes = tv.ops.nms(torch.from_numpy(pred_boxes), torch.from_numpy(pred_scores), iou_threshold=0.5)
        result2 = result2[indexes.numpy()]
        # import epdb; epdb.set_trace()
        pil_image = PIL.Image.fromarray(image)
        for idx in range(result2.shape[0]):
            if result2[idx, -1] > confidence_threshold:
                print(pred_boxes[idx])
                pil_image = draw_rectangle(pil_image, box=pred_boxes[idx], color_hex="#f5428d")
        cols[col].image(im2show, caption=model_name, width=400)
        cols[col+1].image(pil_image, caption=model_name, width=400)


def demo_with_image_fname(image_fname, models):
    image = PIL.Image.open(image_fname)
    image = np.array(image)
    apply_detectors(image, image_fname, models)


def demo_with_sm(sm, bpath, models):
    image_id = int(st.session_state.image_id)
    fname = sm.session.query(Image.filename).filter(Image.id==image_id).first()
    image_filename = f"{bpath}/{fname.filename}"
    demo_with_image_fname(image_filename, models)


def demo_with_url(models):
    image_url = st.session_state.image_url
    import requests
    from io import BytesIO
    res = requests.get(image_url)
    if res.status_code == 200:
        with open("/tmp/test.jpg", "wb") as fid:
            fid.write(res.content)
        demo_with_image_fname("/tmp/test.jpg", models)
    else:
        st.warning(f"{image_url} is invalid")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Compare two detector outputs."
                                     "\nTo get the detector output, please use tools/test.py with --format-only flag.")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Detector config filename.", nargs="+")
    parser.add_argument("--bpath", type=str, help="Path to the base image.",
                        default="/media/nas/private-dataset")
    
    args = parser.parse_args()

    models = {}
    for onnx_fname in args.onnx:
        models[onnx_fname] = load_detector(onnx_fname)

    ## Add threshold sliders.
    confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?',
                                             0.0, 1.0, 0.5, 0.01)

    db_name = st.selectbox(label="database names",
                           options=["ad4", "chictopia", "zara",
                                    "md", "zalando", "asos",
                                    "modcloth", "rentrunway",
                                    "21buttons", "vinted", 
                                    "brands"],
                           index=0)
    sm = SessionManager(db_name, "/")
    image_id = st.text_input("Image id", value=1,
                             help="Select image id",
                             key="image_id",
                             on_change=demo_with_sm,
                             args=(sm, args.bpath, models))
    """
    with st.sidebar.expander("Use research database", expanded=False):
        db_name = st.selectbox(label="database names",
                                       options=["ad4", "chictopia", "zara",
                                                "md", "zalando", "asos",
                                                "modcloth", "rentrunway",
                                                "21buttons"],
                                       index=0)
        sm = SessionManager(db_name, "/")
        image_id = st.text_input("Image id", value=1,
                                 help="Select image id",
                                 key="image_id",
                                 on_change=demo_with_sm,
                                 args=(sm, args.bpath, models))

    with st.sidebar.expander("Use internet images", expanded=False):
        image_url = st.text_input("Image url",
                                  help="Image url",
                                  on_change=demo_with_url,
                                  key="image_url",
                                  args=(models,))
    """
