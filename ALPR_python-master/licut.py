"""CLEANED"""
import numpy as np
from cfour import *
import cv2

def detect_plate_onnx_id(frames, model=plate_detection_model, car_boxes=None):
    out = {}
    car_id = {}
    img = []
    conf = []
    bbox = []
    for j, (id_img, image) in enumerate(frames.items()):
        if car_boxes:
            (x1, y1),(x2,y2) = [i for i in car_boxes[j]]

        image = np.array(image)
        boxes, labels, probs = model.predict(image, 1, prob_threshold=0.4)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            maximum_y,maximum_x,_ = image.shape
            x_min = int(max(0,box[0]))
            x_max = int(min(box[2],maximum_x))
            y_min = int(max(0,box[1]))
            y_max = int(min(box[3],maximum_y))
            # print(y_min, y_max, x_min, x_max)
            cropped = image[y_min:y_max, x_min:x_max]
            if car_boxes:
                bbox.append([(x1 + x_min,y1 + y_min),(x1 + x_max,y1 + y_max)])
            img.append(cropped)
            conf.append(int(probs[i]*100))
        out[id_img] = [img, conf, bbox]
        car_id[id_img] = car_boxes[j]
        img = []
        conf = []
        bbox = [] 
    return out, car_id
