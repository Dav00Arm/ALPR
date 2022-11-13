"""CLEANED"""
import torch
import numpy as np
from prdctr import Predictor_ONNX

image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

def create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu'), onnx=True):
    predictor = Predictor_ONNX(net, image_size, image_mean,
                            image_std,
                            nms_method=nms_method,
                            candidate_size=candidate_size,
                            sigma=sigma,
                            device=device)
    return predictor
