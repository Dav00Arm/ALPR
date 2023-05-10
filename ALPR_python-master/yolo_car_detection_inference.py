import torch
import numpy as np
import torch.backends.cudnn as cudnn
from aug import letterbox
from genr import non_max_suppression, scale_coords
from pslt import box_crop
from configs.model_configs import car_det_configs
from configs.general import general_configs, nms_configs
cudnn.benchmark = True


def car_detection_yolo_one_id(images,
                              model=car_det_configs['car_detection_model'],
                              stride=car_det_configs['stride'],
                              pt=car_det_configs['pt'],
                              imgsz=car_det_configs['imgsz'],
                              device=general_configs['device']):
    # Load model 
    bs = 1  # batch_size
    out = []
    draw_boxes = []
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    images_processed = []
    confs = []
    classes = []

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

        detections = model(im, augment=False, visualize=False)

        # NMS
        detections = non_max_suppression(
            detections, nms_configs['conf_thres'], nms_configs['iou_thres'], car_det_configs['class_ids'],
            max_det=nms_configs['max_det']
        )

        # Process detections
        for i, det in enumerate(detections):
            im0 = images
            if len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                # print(det)
                # print(reversed(det))
                for *xyxy, conf, cls in reversed(det):
                    img = box_crop(im0, xyxy)

                    out.append(img)
                    draw_boxes.append([(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))])
                    classes.append(cls)
                    confs.append(conf)

    return out, draw_boxes, confs, classes
