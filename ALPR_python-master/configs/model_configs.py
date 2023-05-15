import torch
import string
import onnxruntime as ort
from yolo_general_utils import check_img_size
from models_decrypt import decrypt_file, load_graph
from yolo_common import DetectMultiBackend
from ocr_CTC import AttnLabelConverter
from ocr_dataset import MyAlignCollate
from vslite import create_mobilenetv2_ssd_lite_predictor
from configs.general import general_configs

"""Car detection"""
car_detection_weights = "./models/yolov5n_cpu.onnx"# if general_configs['device'].type == 'cpu' else "../models/yolov5n_gpu.onnx"
car_detection_model = DetectMultiBackend(
    weights=decrypt_file(load_graph(car_detection_weights), general_configs['_key']), device=general_configs['device']
)
stride, pt = car_detection_model.stride, car_detection_model.pt
imgsz = check_img_size((320, 320), s=stride)

car_det_configs = {
    'car_detection_model': car_detection_model,
    'class_ids': [2, 3, 5, 7],
    'class_names': {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    },
    'stride': stride,
    'pt': pt,
    'imgsz': imgsz
}

"""Plate detection"""
plate_detection_weights = "../models/plate_detection_cpu.onnx" #if general_configs['device'].type == 'cpu' else "../models/plate_detection_gpu.onnx"
sess_plate = ort.InferenceSession(decrypt_file(load_graph(plate_detection_weights), general_configs['_key']), providers=['CPUExecutionProvider' if general_configs['device'].type == 'cpu' else 'CUDAExecutionProvider'])
plate_detection_model = create_mobilenetv2_ssd_lite_predictor(sess_plate, candidate_size=200, onnx=True, device=general_configs['device'])

plate_det_configs = {
    'plate_detection_model': plate_detection_model,
    'prob_threshold': 0.4
}

"""Craft"""
craft_weights = "../models/craft_cpu.onnx" #if general_configs['device'].type == 'cpu' else "../models/craft_gpu.onnx"
craft_model = ort.InferenceSession(decrypt_file(load_graph(craft_weights), general_configs['_key']), providers=['CPUExecutionProvider' if general_configs['device'].type == 'cpu' else 'CUDAExecutionProvider'])

craft_configs = {
    'craft_model': craft_model,
    'text_threshold': 0.7,
    'link_threshold': 0.4,
    'low_text': 0.4,
    'use_refiner': True,
    'poly': False
}

"""Refiner"""
refiner_weights = "../models/refiner_cpu.onnx"# if general_configs['device'].type == 'cpu' else "../models/refiner_gpu.onnx"
refine_net_model = ort.InferenceSession(decrypt_file(load_graph(refiner_weights), general_configs['_key']), providers=['CPUExecutionProvider' if general_configs['device'].type == 'cpu' else 'CUDAExecutionProvider'])
# cr_rf_params = params_craft_refiner(craft_model, refine_net_model)

refiner_configs = {
    'refine_net_model': refine_net_model,
    'use_poly': False
}

"""Plate recognition"""
# plate_recognition_model = torch.nn.DataParallel(Model()).to(general_configs['device'])
# plate_recognition_model.load_state_dict(torch.load(io.BytesIO(decrypt_file(load_graph(plate_recognition_weights), general_configs['_key'])), map_location=general_configs['device']))

character = string.printable[:-6]

plate_rec_configs = {
    'plate_recognition_weights': '../models/ocr(PN).pth',
    'batch_max_length': 11,
    'converter': AttnLabelConverter(character),
    'plate_recognition_model': torch.jit.load('../models/ocr(PN).pt').to('cpu'),
    'AlignCollate_demo': MyAlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False),
    'mc_address': ['40:8d:5c:c2:9b:55']
}
