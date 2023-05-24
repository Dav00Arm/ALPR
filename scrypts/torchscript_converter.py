import torch# import yaml
# import onnxruntime as ort

import string
from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.CTC import AttnLabelConverter
from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.model import Models

"""Plate recognition"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plate_recognition_weights = 'models/ocr(PN).pth'
batch_max_length = 11
character = string.printable[:-6]

imgH = 32
imgW = 100
converter = AttnLabelConverter(character)

model = torch.nn.DataParallel(Model()).to(device)
model.load_state_dict(torch.load(plate_recognition_weights, map_location=device))

example = torch.rand((1, 1, 32, 100))
model.eval()

traced_script_module = torch.jit.trace(model,(example,))
traced_script_module.save("models/ocr(PN).pt")
