import torch 

import yaml
import onnxruntime as ort

import string
from vslite import create_mobilenetv2_ssd_lite_predictor
from utllscr import AttnLabelConverter
from mdl import Model

from dtst import MyAlignCollate
from cmn import DetectMultiBackend

from genr import check_img_size
# from dtsnd import get_mac
from gtmdls import decrypt_file, load_graph

# import io
from datetime import datetime
"""Camera configurations"""

# ips = ['37.186.127.77']
# admin
# 22dvcgwqw7
camera_urls = ['rtsp://admin:22dvcgwqw7@37.252.72.204:555/cam/realmonitor?channel=1&subtype=0']#,'rtsp://admin:22dvcgwqw7@37.252.72.204:556/cam/realmonitor?channel=1&subtype=0',
                # 'rtsp://admin:22dvcgwqw7@37.252.72.204:557/cam/realmonitor?channel=1&subtype=0','rtsp://admin:25802580Code@37.252.72.204:558/cam/realmonitor?channel=1&subtype=0',
                # 'rtsp://admin:22dvcgwqw7@37.252.72.204:559/cam/realmonitor?channel=1&subtype=0']

# camera_urls = ['rtsp://admin:22dvcgwqw7@192.168.1.221:554/cam/realmonitor?channel=1&subtype=0','rtsp://admin:22dvcgwqw7@192.168.1.222:554/cam/realmonitor?channel=1&subtype=0',
#                 'rtsp://admin:22dvcgwqw7@192.168.1.223:554/cam/realmonitor?channel=1&subtype=0','rtsp://admin:25802580Code@192.168.1.227:554/cam/realmonitor?channel=1&subtype=0',
#                 'rtsp://admin:22dvcgwqw7@192.168.1.225:554/cam/realmonitor?channel=1&subtype=0']
mac_addresses = ['9c:14:63:64:ec:c0','9c:14:63:64:e5:f0','9c:14:63:64:e8:ac','9c:14:63:64:eb:0a','9c:14:63:64:e8:af']

spots_per_camera = [2,2,2,2,2]
locator_url = 'https://gaz.locator.am/api/camera_info'
"""User configurations"""
user_configs = yaml.safe_load(open("user_config.yaml", "r"))

"""General"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_key = "ThisistheonlyKeyThatwillwork123@%^*&((??__+(I_GAS"
"""Car detection"""
car_detection_weights = "models/yolov5n_cpu.onnx" if device.type=='cpu' else "models/yolov5n_gpu.onnx"
car_detection_model =  DetectMultiBackend(weights=decrypt_file(load_graph(car_detection_weights), _key), device=device)
stride, pt = car_detection_model.stride, car_detection_model.pt

imgsz = check_img_size((320, 320), s=stride)
# print(stride,pt,imgsz)
"""Plate detection"""
plate_detection_weights = "models/plate_detection_cpu.onnx" if device.type=='cpu' else "models/plate_detection_gpu.onnx"
sess_plate = ort.InferenceSession(
decrypt_file(load_graph(plate_detection_weights),_key), providers=['CPUExecutionProvider' if device.type=='cpu' else 'CUDAExecutionProvider'])
plate_detection_model = create_mobilenetv2_ssd_lite_predictor(sess_plate, candidate_size=200, onnx=True)

"""Craft"""
craft_weights = "models/craft_cpu.onnx" if device.type=='cpu' else "models/craft_gpu.onnx"
craft_model = ort.InferenceSession(
            decrypt_file(load_graph(craft_weights), _key), providers=['CPUExecutionProvider' if device.type=='cpu' else 'CUDAExecutionProvider'])
text_threshold = 0.7 
link_threshold = 0.4
low_text =0.4
use_refiner = True
poly = False

"""Refiner"""
refiner_weights = "models/refiner_cpu.onnx" if device.type=='cpu' else "models/refiner_gpu.onnx"
refine_net_model = ort.InferenceSession(
            decrypt_file(load_graph(refiner_weights), _key), providers=['CPUExecutionProvider' if device.type=='cpu' else 'CUDAExecutionProvider']) 
# cr_rf_params = params_craft_refiner(craft_model,refine_net_model)
use_poly = False

"""Plate recognition"""
plate_recognition_weights = 'models/ocr(PN).pth' 
batch_max_length = 11
character = string.printable[:-6]
# print(character)
imgH = 32
imgW = 100
converter = AttnLabelConverter(character)
# plate_recognition_model = torch.nn.DataParallel(Model()).to(device)
# plate_recognition_model.load_state_dict(torch.load(io.BytesIO(decrypt_file(load_graph(plate_recognition_weights), _key)), map_location=device)) 
plate_recognition_model = torch.jit.load('models/ocr(PN).pt').to(device)
AlignCollate_demo = MyAlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=False)
# plate_recognition_model.eval()
mc_address = ['40:8d:5c:c2:9b:55']