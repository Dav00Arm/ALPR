import torch

# import yaml
# import onnxruntime as ort

import string
# from vslite import create_mobilenetv2_ssd_lite_predictor
from utllscr import AttnLabelConverter
from mdl import Model
from configs.general import general_configs

from dtst import MyAlignCollate
# from cmn import DetectMultiBackend

# from genr import check_img_size
# from dtsnd import get_mac
from gtmdls import decrypt_file, load_graph

import io
from datetime import datetime

"""Plate recognition"""
_key = "ThisistheonlyKeyThatwillwork123@%^*&((??__+(I_GAS"

plate_recognition_weights = 'models/ocr(PN).pth'
batch_max_length = 11
character = string.printable[:-6]
# print(character)
imgH = 32
imgW = 100
converter = AttnLabelConverter(character)

# plate_recognition_model = torch.nn.DataParallel(Model()).to(general_configs['device'])
# plate_recognition_model.load_state_dict(torch.load(io.BytesIO(decrypt_file(load_graph(plate_recognition_weights), _key)), map_location=device))
# # plate_recognition_model = torch.jit.load('./traced_ocr_model_fixed.pt').to(general_configs['device'])
# AlignCollate_demo = MyAlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=False)
# plate_recognition_model.eval()

model = torch.nn.DataParallel(Model()).to(general_configs['device'])
# model.load_state_dict(torch.load(io.BytesIO(decrypt_file(load_graph(plate_recognition_weights), _key)), map_location=general_configs['device']))
model.load_state_dict(torch.load(plate_recognition_weights, map_location=general_configs['device']))
# # example = cv2.imread("/home/user/deep-text-recognition-benchmark/demo_image/demo_1.png").to(general_configs['device'])

example = torch.rand((1,1,32,100))
model.eval()
# example = example[0][0].fill_(0.2)
# example = example[0][1].fill_(0.2)


# text_for_pred = torch.LongTensor(1, 12).fill_(0).to(general_configs['device'])

# out = model(example,text_for_pred)
# _, preds_index = out.max(2)
# print(preds_index)
# example = cv

# # model(example,text_for_pred)
traced_script_module = torch.jit.trace(model,(example,))
traced_script_module.save("models/ocr(PN).pt")
