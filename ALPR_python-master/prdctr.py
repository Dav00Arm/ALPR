"""CLEANED"""
import torch
import numpy as np
import bu
from dproc import PredictionTransform
import time

class Predictor_ONNX:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        # self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        # For SSD only
        self.input_name = self.net.get_inputs()[0].name
        self.ouputs_name_1 = self.net.get_outputs()[0].name
        self.ouputs_name_2 = self.net.get_outputs()[1].name
        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def predict(self, image, top_k=-1, prob_threshold=0.3):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        results = self.net.run([self.ouputs_name_1,self.ouputs_name_2],{self.input_name:np.array(images)})
        scores, boxes = torch.Tensor(results[0]), torch.Tensor(results[1])        
        boxes = boxes[0]
        scores = scores[0]

        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]

            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]

            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = bu.hard_nms(box_probs, 
                                        top_k=top_k,
                                        candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]