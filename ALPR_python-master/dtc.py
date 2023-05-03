from matplotlib.pyplot import draw
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from cfour import *
from aug import letterbox
from genr import non_max_suppression, scale_coords
from pslt import box_crop
cudnn.benchmark = True


def car_detection_yolo_one(images, model=car_detection_model, stride=stride, pt=pt, imgsz=imgsz, device=device):
    # Load model 
    bs = 1  # batch_size
    out = []
    draw_boxes = []
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    images_processed = []
    confd = []
    classes = []
    for img in images:
        # print(img.shape)
        img = letterbox(img, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        images_processed.append(img)

    for j, im in enumerate(images_processed):
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, 0.45, 0.45, [2,3,5,7], False, max_det=1)
        # print(len(pred))  # [[],[],[]]
        # Process predictions
        # for i, det in enumerate(pred):  # per image
        det = pred[0]
        im0 = images[j]
        # print(len(det))
        if len(det) > 0:
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                img = box_crop(im0, xyxy)
                # print(type(out[j]))
                out.append(img)
                # print(out[j])
                draw_boxes.append([(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3]))])
                classes.append("Car")
                confd.append(conf)
        else:
            out.append(np.array([]))

    return out, draw_boxes, confd, classes

def car_detection_yolo_one_id(images, model=car_detection_model, stride=stride, pt=pt, imgsz=imgsz, device=device):
    # Load model 
    bs = 1  # batch_size
    out = []
    draw_boxes = []
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    images_processed = []
    confd = []
    classes = []
    # for img in images:
        # print(img.shape)
    img = letterbox(images, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    images_processed.append(img)

    for j, im in enumerate(images_processed):
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # print("*"*10)
        pred = model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, 0.40, 0.5, [2,3,5,7], False, max_det=100)
        # print(pred[0].shape)
        # print("*"*10)
        # print()
        # print(len(pred))  # [[],[],[]]
        # Process predictions
        for i, det in enumerate(pred):  # per imageconf
            im0 = images
            # print(len(det))
            # print(type(det))
            # print(len(det))
            if len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                # print(det)
                # print(reversed(det))
                for *xyxy, conf, cls in reversed(det):
                    img = box_crop(im0, xyxy)

                    out.append(img)
                    draw_boxes.append([(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3]))])
                    classes.append(0)
                    confd.append(conf)

    return out, draw_boxes, confd, classes