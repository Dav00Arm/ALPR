# import numpy
# import onnxruntime as rt
# import cv2
# import numpy as np
# from torch.autograd import Variable

# import torch.onnx
# import torchvision
# import torch
# from utils import copyStateDict
# from PROJECT_Text_Detection_Model_SOFTWARE_AI.craft import CRAFT
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.model import Model
# from __init__ import *
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.utils import CTCLabelConverter
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.utils import AttnLabelConverter
# from PROJECT_Text_Detection_Model_SOFTWARE_AI.refinenet import RefineNet
# import PROJECT_Text_Detection_Model_SOFTWARE_AI.imgproc as imgproc
# from craft.utils import test_net
# import time
##############################
# canvas_size = 100
# image = cv2.imread('/home/user/ALPR/inference/init_image/plate.jpg')
# """REFINER.ONNX"""
# sess = rt.InferenceSession(
#     "craft_.onnx", providers=['CPUExecutionProvider'])
# print(rt.get_available_providers())
# img_resized, ratio_h,ratio_w = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
# img_resized = cv2.resize(image,( canvas_size, canvas_size),cv2.INTER_AREA)

# preprocessing
# input_name1 = sess.get_inputs()[0].name
# input_name2 = sess.get_inputs()[1].name
# label_name_1 = sess.get_outputs()[0].name
# for _ in range(10):
#     start = time.time()
#     # x = imgproc.normalizeMeanVariance(img_resized)
#     # x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
#     # x = Variable(x.unsqueeze(0))  
#     map = sess.run([label_name_1], {input_name1:np.array(image).astype(numpy.float32)})
#     end = time.time()
#     print("ONNX: ", end - start)
# print(map[0].shape)
# print(feature[0].shape)

# print(len(pred_onnx))
"""CRAFT.PTH"""
# import torch.backends.cudnn as cudnn
# craft_pth = CRAFT()

# craft_pth.load_state_dict(copyStateDict(torch.load(craft_path)))
# craft_pth = craft_pth.cuda()
# craft_pth = torch.nn.DataParallel(craft_pth)
# cudnn.benchmark = False
# # img_resized = cv2.resize(image,( canvas_size, canvas_size),cv2.INTER_AREA)

# for _ in range(10):
    # test_net(craft_pth,image,text_threshold,link_threshold,low_text,cuda=False,poly=True,refine_net=None,canvas_size=canvas_size,mag_ratio=1.5)
# print(text_for_pred)

"""CONVERTER"""
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.model import Model
# from __init__ import *
# import torch 
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.utils import CTCLabelConverter
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.utils import AttnLabelConverter
# from torch.autograd import Variable
# opt = argument()
# if 'CTC' in opt.Prediction:
#     converter = CTCLabelConverter(opt.character)
# else:
#     converter = AttnLabelConverter(opt.character)
# opt.num_class = len(converter.character)

# if opt.rgb:
#     opt.input_channel = 3
# net = Model(opt)

# net = torch.nn.DataParallel(net).to('cpu')

# net.load_state_dict(torch.load('models/ocr.pth', map_location='cpu'))
# dummy_input1 = Variable(torch.randn([1, 1, 32, 100]))
# dummy_input2 = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to('cpu')
# # dummy_input2 = Variable(torch.randn([1, 32, 50, 50])).cuda()
# input_names = ['image', 'text_pred']
# output_names = ['preds']
# torch.onnx.export(net.module, (dummy_input1, dummy_input2), "ocr.onnx", input_names=input_names,output_names=output_names,opset_version=11)


# """REFINER.PTH"""

# import torch.backends.cudnn as cudnn

# craft_pth = CRAFT()

# craft_pth.load_state_dict(copyStateDict(torch.load(craft_path)))
# craft_pth = craft_pth.cuda()
# craft_pth = torch.nn.DataParallel(craft_pth)
# cudnn.benchmark = False


# refine_net = RefineNet()
# refine_net_path = 'models/refiner.pth'
# refine_net.load_state_dict(copyStateDict(torch.load(refine_net_path, map_location='cpu')))
# refine_net = refine_net.cpu()
# refine_net = torch.nn.DataParallel(refine_net)

# for _ in range(10):
#     y, feature = test_net(craft_pth,image,text_threshold,link_threshold,low_text,cuda=False,poly=True,refine_net=refine_net,canvas_size=canvas_size,mag_ratio=1.5)

"""ONNX TO TRT"""

# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import tensorrt as trt
 
# logger to capture errors, warnings, and other information during the build and inference phases
# TRT_LOGGER = trt.Logger()
# print(TRT_LOGGER)
# def build_engine(onnx_file_path):
#     # initialize TensorRT engine and parse ONNX model
#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network()
#     parser = trt.OnnxParser(network, TRT_LOGGER)
#     explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     # parse ONNX
#     with open(onnx_file_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         parser.parse(model.read())
#         print(parser.parse(model.read()))
#     print('Completed parsing of ONNX file')
# # allow TensorRT to use up to 1GB of GPU memory for tactic selection
#     builder.max_workspace_size = 1 << 30
#     # we have only one image in batch
#     builder.max_batch_size = 1
#     # use FP16 mode if possible
#     if builder.platform_has_fast_fp16:
#         builder.fp16_mode = True

#     last_layer = network.get_layer(network.num_layers - 1)
#     # Check if last layer recognizes it's output
#     if not last_layer.get_output(0):
#         # If not, then mark the output using TensorRT API
#         network.mark_output(last_layer.get_output(0))

#     print('Building an engine...')
#     engine = builder.build_cuda_engine(network)
#     context = engine.create_execution_context()
#     print("Completed creating Engine")
 
#     return engine, context

# engine, context = build_engine('models/craft.onnx')


"""####################################################### ONNX TO TENSORRT ####################################################################"""
# def onnx2trt(onnx_file_path='models/craft.onnx'):
#     G_LOGGER = trt.Logger(trt.Logger.WARNING)
#     explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
#         builder.max_batch_size = 100
#         builder.max_workspace_size = 1 << 20

#         print('loading ONNX')
#         with open(onnx_file_path, 'rb') as model:
#             print('beginning ONNX parsing')
#             parser.parse(model.read())
#             print(parser.parse(model.read()))
#         last_layer = network.get_layer(network.num_layers-1)
#         # network.mark_output(last_layer.get_output(0))
#         print('Completed parsing of onnx')
#         print('building the engine')
#         engine = builder.build_cuda_engine(network)
#         print('completed creating the engine')

#         with open('craft.engine', 'wb') as f:
#             f.write(engine.serialize())
#         return engine
# onnx2trt()
"""####################################################### ONNX TO TENSORRT ####################################################################"""


# import tensorrt as trt
# import numpy as np
# import os

# import pycuda.driver as cuda
# import pycuda.autoinit



# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#     def __repr__(self):
#         return self.__str__()

# class TrtModel:
    
#     def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
#         self.engine_path = engine_path
#         self.dtype = dtype
#         self.logger = trt.Logger(trt.Logger.WARNING)
#         self.runtime = trt.Runtime(self.logger)
#         self.engine = self.load_engine(self.runtime, self.engine_path)
#         self.max_batch_size = max_batch_size
#         self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
#         self.context = self.engine.create_execution_context()

                
                
#     @staticmethod
#     def load_engine(trt_runtime, engine_path):
#         trt.init_libnvinfer_plugins(None, "")             
#         with open(engine_path, 'rb') as f:
#             engine_data = f.read()
#         engine = trt_runtime.deserialize_cuda_engine(engine_data)
#         return engine
    
#     def allocate_buffers(self):
        
#         inputs = []
#         outputs = []
#         bindings = []
#         stream = cuda.Stream()
        
#         for binding in self.engine:
#             size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
#             host_mem = cuda.pagelocked_empty(size, self.dtype)
#             device_mem = cuda.mem_alloc(host_mem.nbytes)
            
#             bindings.append(int(device_mem))

#             if self.engine.binding_is_input(binding):
#                 inputs.append(HostDeviceMem(host_mem, device_mem))
#             else:
#                 outputs.append(HostDeviceMem(host_mem, device_mem))
        
#         return inputs, outputs, bindings, stream
       
            
#     def __call__(self,x:np.ndarray,batch_size=2):
        
#         x = x.astype(self.dtype)
        
#         np.copyto(self.inputs[0].host,x.ravel())
        
#         for inp in self.inputs:
#             cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
#         self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
#         for out in self.outputs:
#             cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
#         self.stream.synchronize()
#         return [out.host.reshape(batch_size,-1) for out in self.outputs]



# batch_size = 1
# trt_engine_path = os.path.join("models","craft.engine")
# model = TrtModel(trt_engine_path)
# shape = model.engine.get_binding_shape(0)


# data = np.array(torch.randn([1, 3, 100, 100]))
# result = model(data,batch_size)
# print(result[0].shape,result[1].shape)
    

"""-----------------------------------------------------CAR DETECTION----------------------------------------------------------------------"""
# import jetson.utils
# import jetson.inference
# import numpy as np
# import cv2
# import time

# net = jetson.inference.detectNet('ssd-mobilenet-v2',threshold=0.5)
# cap = cv2.VideoCapture('rtsp://admin:d12345678@192.168.2.64:554/Streaming/Channels/101/')
# ret, orig_image = cap.read()

# def convert_xywh(d):
#     '''
#     Function to convert jetson outputs to list
#     '''
#     x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
#     x, y, w, h = x1, y1, x2 - x1, y2 - y1
#     return np.array([x,y,w,h]) 

# def car_detection_jetson(frame, net):
#     ''' 
#     Detect cars in the image with Jetson Inference
#     :param frame: Image for detection
#     :returns: Cropped image of the car in the image 
#     '''
#     bboxes = []
#     names = []
#     scores = []
#     imgCuda = jetson.utils.cudaFromNumpy(frame)
#     detection = net.Detect(imgCuda) 
#     crop_img = []
#     for d in detection:
#         className = net.GetClassDesc(d.ClassID)
#         if className == 'car' or className == 'bus' or className == "truck":
#             x1_,y1_,w1_,h1_ = [int(i) for i in convert_xywh(d)]

#             bboxes.append(convert_xywh(d))
#             names.append(className)
#             scores.append(d.Confidence)

#             maximum_y,maximum_x,_ = frame.shape
#             x_min = max(0,x1_)
#             x_max = min(x1_+w1_,maximum_x)
#             y_min = max(0,y1_)
#             y_max = min(y1_+h1_,maximum_y)
#             crop_img.append(frame[y_min:y_max,x_min:x_max])
#             cv2.imwrite(str(time.time())+'.jpg', frame[y_min:y_max,x_min:x_max])


#     return crop_img

# car_detection_jetson(orig_image, net)

"""-----------------------------------------------------CAR DETECTION----------------------------------------------------------------------"""

"""-----------------------------------------------------CRAFT ONNX-------------------------------------------------------------------------"""
# from configparser import Interpolation
# import onnxruntime as ort
# import numpy as np
# import cv2
# import torch
# from torch.autograd import Variable

# sess_refiner = ort.InferenceSession(
# "models/refiner.onnx", providers=['CPUExecutionProvider'])

# # image = cv2.imread('/home/user/ALPR/inference/init_image/plate.jpg')
# # image = cv2.resize(image, (100,100), interpolation=cv2.INTER_AREA)


# input1 = np.array(torch.randn([50, 50, 2]))
# input2 = np.array(torch.randn([32, 50, 50]))
# input_name1 = sess_refiner.get_inputs()[0].name
# input_name2 = sess_refiner.get_inputs()[1].name
# label_name1 = sess_refiner.get_outputs()[0].name
# pred_onx = sess_refiner.run([label_name1], {input_name1: [input1.astype(np.float32)], input_name2:[input2.astype(np.float32)]})

# print(len(pred_onx[0][0]))

# torch.Size([1, 50, 50, 2])
# torch.Size([1, 32, 50, 50])
"""-----------------------------------------------------------------------------------------------------------------------------------------"""
# import argparse
# import json
# import os
# import platform
# import subprocess
# import sys
# import time
# import warnings
# from pathlib import Path

# import pandas as pd
# import torch
# import yaml
# from torch.utils.mobile_optimizer import optimize_for_mobile

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# if platform.system() != 'Windows':
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from yolov5.models.experimental import attempt_load
# from yolov5.models.yolo import Detect
# from yolov5.utils.dataloaders import LoadImages
# from yolov5.utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_version, colorstr,
#                            file_size, print_args, url2file)
# from yolov5.utils.torch_utils import select_device
# import onnx

# def export_onnx(model, im, file, opset, train, dynamic, prefix=colorstr('ONNX:')):
#     # YOLOv5 ONNX export
#     try:
#         check_requirements(('onnx',))
#         f = file.with_suffix('.onnx')
#         torch.onnx.export(
#             model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
#             im.cpu() if dynamic else im,
#             f,
#             verbose=False,
#             opset_version=opset,
#             training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
#             do_constant_folding=not train,
#             input_names=['images'],
#             output_names=['output'],
#             dynamic_axes={
#                 'images': {
#                     0: 'batch',
#                     2: 'height',
#                     3: 'width'},  # shape(1,3,640,640)
#                 'output': {
#                     0: 'batch',
#                     1: 'anchors'}  # shape(1,25200,85)
#             } if dynamic else None)

#         # Checks
#         model_onnx = onnx.load(f)  # load onnx model
#         onnx.checker.check_model(model_onnx)  # check onnx model

#         # Metadata
#         d = {'stride': int(max(model.stride)), 'names': model.names}
#         for k, v in d.items():
#             meta = model_onnx.metadata_props.add()
#             meta.key, meta.value = k, str(v)
#         onnx.save(model_onnx, f)
#         return f
#     except Exception as e:
#         LOGGER.info(f'{prefix} export failure: {e}')


'''---------------------------------------------------------------------------------------'''

# from misc import VideoCaptureThreading
# from yolov5.detect import car_detection_yolo, car_detection_yolo_one
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.torch_utils import select_device
# from yolov5.utils.general import check_img_size
# import time
# url1 = 'rtsp://admin:d12345678@192.168.2.64:554/Streaming/Channels/101/'
# cap = VideoCaptureThreading(url1)
# cap.start()
# url2 = 'rtsp://Traffic:A7Karen*@192.168.2.222:554/Streaming/Channels/101/'
# cap2 = VideoCaptureThreading(url2)
# cap2.start()
# device = select_device('cpu')
# car_detection_model = DetectMultiBackend(weights='./models/yolov5n_aranc.onnx', device=device, dnn=False, data=None, fp16=False)
# # car_detection_model = DetectMultiBackend(weights='./models/yolov5n_dynamic.onnx', device=device, dnn=False, data=None, fp16=False)
# stride, pt = car_detection_model.stride, car_detection_model.pt
# imgsz = check_img_size((320, 320), s=stride)
# for frame1, frame2 in zip(cap.generator(), cap2.generator()):
#     start = time.time()
#     # car_images = car_detection_yolo(car_detection_model,stride,pt,imgsz,[frame1,frame2])
#     car_images = car_detection_yolo_one(car_detection_model,stride,pt,imgsz,[frame1,frame2])
#     print(time.time() - start)

'''---------------------------------------------------------------------------------------'''
# import torch 
# from craft.craft import CRAFT
# from utils import copyStateDict
# from torch.autograd import Variable
# import numpy as np
# craft_path = './models/craft.pth'
# craft  = CRAFT()
# craft.load_state_dict(copyStateDict(torch.load(craft_path, map_location='cpu')))
# dummy_input1 = torch.randn([1, 3, 100, 100])
# input_names = ['image']
# output_names = ['y', 'feature']

# torch.onnx.export(craft, (dummy_input1), "craft.onnx", input_names=input_names,output_names=output_names,opset_version=11)

'''---------------------------------------------------------------------------------------'''
# import cv2
# import torch
# import numpy as np
# import onnxruntime as ort
# from craft import craft.utils
# from craft import imgproc
# from torch.autograd import Variable
# import time

# text_threshold = 0.7 
# link_threshold = 0.4
# low_text =0.4

# # Proccessings
# def craft_processings(image,canvas_size=100):
#     img_resized, ratio_h,ratio_w = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
#     img_resized = cv2.resize(image,( canvas_size, canvas_size),cv2.INTER_AREA)
#     # ratio = args.canvas_size / max(height, width) 
#     ratio_h = 1 / ratio_h
#     ratio_w = 1 / ratio_w

#     # preprocessing
#     x = imgproc.normalizeMeanVariance(img_resized)
#     x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
#     x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
#     return np.array(x).astype(np.float32),ratio_h,ratio_w

# def cropping_image(img,pts):
#     '''
#     Crop image based on poly coordinates
#     :param img: Image to be cropped
#     :param pts: Polys for cropping
#     :returns : Cropped image
#     '''
#     maximum_y,maximum_x,_ = img.shape
#     rect = cv2.boundingRect(pts)
#     x,y,w,h = rect
#     x_min = max(0,x)
#     x_max = min(x+w,maximum_x)
#     y_min = max(0,y)
#     y_max = min(y+h,maximum_y)
#     cropped = img[y_min:y_max, x_min:x_max].copy()
#     pts = pts - pts.min(axis=0)
#     mask = np.zeros(cropped.shape[:2], np.uint8)
#     cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
#     dst = cv2.bitwise_and(cropped, cropped, mask=mask)
#     return dst

# image_path = './example_images/plate_pred.jpg'
# image = cv2.imread(image_path)
# x,ratio_h, ratio_w = craft_processings(image) 
# net_craft = ort.InferenceSession(
#             "./models/craft.onnx", providers=['CPUExecutionProvider'])
# net_refiner = ort.InferenceSession(
#             "./models/refiner.onnx", providers=['CPUExecutionProvider'])
# cr_input_name1 = net_craft.get_inputs()[0].name
# cr_label_name1 = net_craft.get_outputs()[0].name
# cr_label_name2 = net_craft.get_outputs()[1].name

# ref_input_name1 = net_refiner.get_inputs()[0].name
# ref_input_name2 = net_refiner.get_inputs()[1].name
# ref_label_name1 = net_refiner.get_outputs()[0].name
# # ref_label_name2 = net_refiner.get_outputs()[1].name
# print('*'*10,"CRAFT",'*'*10)
# print(cr_input_name1,cr_label_name1,cr_label_name2)
# print('*'*10,"REFINER",'*'*10)
# print(ref_input_name1,ref_input_name2,ref_label_name1)
# start = time.time()
# y, feature = net_craft.run([cr_label_name1,cr_label_name2], {cr_input_name1: np.array(x).astype(np.float32)})
# print("y: ",y.shape)
# print("feature : ",feature.shape)
# y_refiner = net_refiner.run([ref_label_name1],{ref_input_name1: np.array(y), ref_input_name2: np.array(feature)})
# print(time.time() - start)
# # y = y[0]
# y_refiner = y_refiner[0]
# score_text = y[0,:,:,0]
# score_link = y_refiner[0,:,:,0]

# # Post-processing
# # print("aaaaaaaaaaaaaa") 
# boxes, polys = craft.utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, use_poly=True)

# # coordinate adjustment
# boxes = craft.utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

# polys = craft.utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
# for k in range(len(polys)):
#     if polys[k] is None: polys[k] = boxes[k]

'''---------------------------------------------------------------------------------------'''

# a = []  
# for i in range(3):
#     a.append([])
#     for j in range(4):
#         a[i].append(j)
# print(a)
'''---------------------------------------------------------------------------------------'''
# import numpy as np
# import torch
# array1 = torch.rand(28,28,3)
# array2 = torch.rand(28,)
# a = [np.array(array1),np.array(array2)]
# np.array(a)
'''---------------------------------------------------------------------------------------'''
# import torch.nn as nn

# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.modules.transformation import TPS_SpatialTransformerNetwork
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.modules.sequence_modeling import BidirectionalLSTM
# from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.modules.prediction import Attention


# class Model(nn.Module):

#     def __init__(self):
#         super(Model, self).__init__()
#         self.stages = {'Trans': 'TPS', 'Feat':'ResNet',
#                        'Seq': 'BiLSTM', 'Pred': 'Attn'}

#         self.Transformation = TPS_SpatialTransformerNetwork(
#             F=20, I_size=(32, 100), I_r_size=(32, 100), I_channel_num=1)


#         self.FeatureExtraction = ResNet_FeatureExtractor(1, 1)

#         self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
#         self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

#         """ Sequence modeling"""
#         self.SequenceModeling = nn.Sequential(
#             BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
#             BidirectionalLSTM(256, 256, 256))
#         self.SequenceModeling_output = 256

#         """ Prediction """

#         self.Prediction = Attention(self.SequenceModeling_output, 256, 96)

#     def forward(self, input, text, is_train=True):
#         """ Transformation stage """
#         if not self.stages['Trans'] == "None":
#             input = self.Transformation(input)

#         """ Feature extraction stage """
#         visual_feature = self.FeatureExtraction(input)
#         visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
#         visual_feature = visual_feature.squeeze(3)

#         """ Sequence modeling stage """
#         contextual_feature = self.SequenceModeling(visual_feature)

#         """ Prediction stage """
#         prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=11)

#         return prediction

# bb1, bb2 = [41, 399, 496, 277], [28, 341, 567, 359]
# bb1, bb2 = [bb1[0], bb1[1], bb1[0]+bb1[2], bb1[1]+bb1[3]], [bb2[0], bb2[1], bb2[0]+bb2[2], bb2[1]+bb2[3]]

# import numpy as np
# import time
# import cv2

# def box_inter_union(arr1, arr2):
#     area1 = (arr1[:, 2] - arr1[:, 0]) * (arr1[:, 3] - arr1[:, 1])
#     area2 = (arr2[:, 2] - arr2[:, 0]) * (arr2[:, 3] - arr2[:, 1])

#     # Intersection
#     top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
#     bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
#     wh = bottom_right - top_left
#     # clip: if boxes not overlap then make it zero
#     intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

#     # union = area1 + area2 - intersection
#     # iou = intersection / union
#     if intersection > 0:return 'Busy'
#     return 'Free'


# image = np.zeros((1000, 1000, 3))
# bb1 = cv2.selectROI('image',image)
# bb1 = [bb1[0], bb1[1], bb1[0]+bb1[2], bb1[1]+bb1[3]]
# cv2.rectangle(image, (bb1[0], bb1[1]),(bb1[2], bb1[3]), (200,0,50), 1)
# bb2 = cv2.selectROI('image',image)
# bb2 = [bb2[0], bb2[1], bb2[0]+bb2[2], bb2[1]+bb2[3]]
# cv2.rectangle(image, (bb2[0], bb2[1]),(bb2[2], bb2[3]), (0,150,150), 1)
# start = time.time()
# print(box_inter_union(np.array([bb1]), np.array([bb2]))
# print('time:', time.time()-start)
# cv2.imshow('image', image)
# cv2.waitKey(0)
"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

# import torch.onnx 

# #Function to Convert to ONNX 
# def Convert_ONNX(model,dummy_input): 

#     # set the model to inference mode 
#     model.eval() 

#     # Let's create a dummy input tensor  
#     # dummy_input = torch.randn(1, input_size, requires_grad=True)  

#     # Export the model   
#     torch.onnx.export(model,         # model being run 
#          dummy_input,       # model input (or a tuple for multiple inputs) 
#          "/home/user/Desktop/models/ocr_best.onnx",       # where to save the model  
#          export_params=True,  # store the trained parameter weights inside the model file 
#          opset_version=16,    # the ONNX version to export the model to 
#         #  do_constant_folding=True,  # whether to execute constant folding for optimization 
#          input_names = ['input1'],   # the model's input names 
#          output_names = ['output']) # the model's output names 
#     print(" ") 
#     print('Model has been converted to ONNX') 

# _key = "ThisistheonlyKeyThatwillwork123@%^*&((??__+(I_GAS"
# plate_recognition_model = torch.nn.DataParallel(Model()).to(device)
# plate_recognition_model.load_state_dict(torch.load(io.BytesIO(decrypt_file(load_graph(plate_recognition_weights), _key)), map_location=device))
# example = torch.rand((1,1,32,100))
# text_for_pred = torch.zeros(1, 12,dtype=torch.float64)
# import numpy as np
# Convert_ONNX(plate_recognition_model.module,example)

from VideoCapture import VideoCaptureThreading
import cv2
video_captures = []
cap = VideoCaptureThreading(0)
cap.start()
# threads.append(cap)
video_captures.append(cap.generator())
for frames in zip(*video_captures):
    cv2.imshow("Image", frames[0])
    cv2.waitKey(1)
