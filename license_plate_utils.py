import numpy as np
from PROJECT_Plate_Detection_Model_SOFTWARE_AI.plate_det_configs import plate_det_configs


def convert_xywh(d):
    """
    Function to convert jetson outputs to list
    """
    x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    return np.array([x, y, w, h])


def detect_plate_onnx(frames, model=plate_det_configs['plate_detection_model']):
    img = []
    conf = []
    for image in frames:
        image = np.array(image)
        cropped = image
        pred_conf = 0
        if len(image):
            boxes, labels, probs = model.predict(image, 1, prob_threshold=plate_det_configs['prob_threshold'])
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                maximum_y, maximum_x, _ = image.shape
                x_min = int(max(0, box[0]))
                x_max = int(min(box[2], maximum_x))
                y_min = int(max(0, box[1]))
                y_max = int(min(box[3], maximum_y))
                
                cropped = image[y_min:y_max, x_min:x_max]
                pred_conf = int(probs[i]*100)
        img.append(cropped)
        conf.append(pred_conf)
    return img, conf


def detect_plate_onnx_id(frames, model=plate_det_configs['plate_detection_model'], car_boxes=None):
    out = {}
    car_id = {}
    img = []
    conf = []
    bbox = []
    for j, (id_img, image) in enumerate(frames.items()):
        if car_boxes:
            (x1, y1), (x2, y2) = [i for i in car_boxes[j]]

        image = np.array(image)
        boxes, labels, probs = model.predict(image, 1, prob_threshold=plate_det_configs['prob_threshold'])
        for i in range(boxes.size(0)):
            box = boxes[i, :]

            maximum_y, maximum_x, _ = image.shape

            x_min = int(max(0, box[0]))
            x_max = int(min(box[2], maximum_x))
            y_min = int(max(0, box[1]))
            y_max = int(min(box[3], maximum_y))
            
            cropped = image[y_min: y_max, x_min: x_max]

            if car_boxes:
                bbox.append([(x1 + x_min, y1 + y_min), (x1 + x_max, y1 + y_max)])

            img.append(cropped)
            conf.append(int(probs[i] * 100))

        out[id_img] = [img, conf, bbox]
        car_id[id_img] = car_boxes[j]
        img = []
        conf = []
        bbox = []

    return out, car_id


def plate_detection(net, frames):
    """
    Detect plates in the image with Jetson Inference
    :param frame: Image for detection
    :returns: Cropped image of the license image in the image
    """

    bboxes = []
    names = []
    scores = []
    crop_img = []
    for frame in frames:
        imgCuda = jetson.utils.cudaFromNumpy(frame)
        detection = net.Detect(imgCuda) 
        for d in detection:
            className = net.GetClassDesc(d.ClassID)
            if className == 'plate':
                x1_, y1_, w1_, h1_ = [int(i) for i in convert_xywh(d)]

                bboxes.append(convert_xywh(d))
                names.append(className)
                scores.append(d.Confidence)

                maximum_y, maximum_x, _ = frame.shape
                x_min = max(0, x1_)
                x_max = min(x1_ + w1_, maximum_x)
                y_min = max(0, y1_)
                y_max = min(y1_ + h1_, maximum_y)
                crop_img.append(frame[y_min: y_max, x_min: x_max])
        
    return crop_img
