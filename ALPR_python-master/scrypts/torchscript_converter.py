import torch# import yaml
# import onnxruntime as ort

import string
from ocr.CTC import AttnLabelConverter
from ocr.model import Model
from configs.general import general_configs

"""Plate recognition"""

plate_recognition_weights = 'models/ocr(PN).pth'
batch_max_length = 11
character = string.printable[:-6]

imgH = 32
imgW = 100
converter = AttnLabelConverter(character)

model = torch.nn.DataParallel(Model()).to(general_configs['device'])
model.load_state_dict(torch.load(plate_recognition_weights, map_location=general_configs['device']))

example = torch.rand((1, 1, 32, 100))
model.eval()

traced_script_module = torch.jit.trace(model,(example,))
traced_script_module.save("models/ocr(PN).pt")
