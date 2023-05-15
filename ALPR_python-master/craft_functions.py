import time
import torch
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np
import craft_utils
import craft_normalize
from utils import params_craft_refiner
from configs.model_configs import craft_configs, refiner_configs


def craft_preprocess(image,canvas_size=100):
    img_resized, ratio_h,ratio_w = craft_normalize.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
    img_resized = cv2.resize(image,( canvas_size, canvas_size),cv2.INTER_AREA)
    ratio_h = 1 / ratio_h
    ratio_w = 1 / ratio_w

    # preprocessing
    x = craft_normalize.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  
    return x, ratio_h, ratio_w


def get_boxes(score_text, score_link, text_threshold, link_threshold, low_text, poly, ratio_w, ratio_h):
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]
    return boxes, polys


def predict_craft(image,
                  net=craft_configs['craft_model'],
                  text_threshold=craft_configs['text_threshold'],
                  link_threshold=craft_configs['link_threshold'],
                  low_text=craft_configs['low_text'],
                  poly=craft_configs['poly'],
                  refine_net=refiner_configs['refine_net_model'],
                  canvas_size=100,
                  mode='onnx',
                  params=params_craft_refiner(craft_configs['craft_model'], refiner_configs['refine_net_model'])):
    """
    CRAFT inference
    """

    x, ratio_h, ratio_w = craft_preprocess(image,canvas_size=canvas_size)

    if mode == 'gpu': 
        x = x.cuda()
        with torch.no_grad():
            y, feature = net(x)
            y_refiner = refine_net(y, feature)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    elif mode == 'onnx':
        y, feature = net.run([params['craft'][1], params['craft'][2]],
                             {params['craft'][0]: np.array(x).astype(np.float32)})
        y_refiner = refine_net.run([params['refine'][2]],
                                   {params['refine'][0]: np.array(y), params['refine'][1]: np.array(feature)})
        y_refiner = y_refiner[0]
        score_text = y[0, :, :, 0]
        score_link = y_refiner[0, :, :, 0]

    # TODO
    elif mode == 'trt':
        pass

    boxes, polys = get_boxes(score_text, score_link, text_threshold, link_threshold, low_text, poly, ratio_w, ratio_h)

    return boxes, polys 


def cropping_image(img, pts):
    """
    Crop image based on poly coordinates
    :param img: Image to be cropped
    :param pts: Polys for cropping
    :returns : Cropped image
    """
    maximum_y, maximum_x, _ = img.shape
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    x_min = max(0, x)
    x_max = min(x + w, maximum_x)
    y_min = max(0, y)
    y_max = min(y + h, maximum_y)
    cropped = img[y_min: y_max, x_min: x_max].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)

    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(cropped, cropped, mask=mask)

    return dst


def crop_lines(images):
    """
    Function for cropping lines in license plate images
    """
    all_results = []
    t = time.time()
    for image in images:
        if len(image) == 0:
            all_results.append([])
        else:   
            boxes, polys = predict_craft(image)
            cropped = []                     
            for poly in polys:
                poly = np.array(poly, dtype=np.int32)
                cropped.append(Image.fromarray(cropping_image(image, poly)).convert('L'))
            all_results.append(cropped)

    return all_results
