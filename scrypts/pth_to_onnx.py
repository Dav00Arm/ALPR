import torch
import onnx
from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.model import Model

model = torch.nn.DataParallel(Model())
# model = Model()
model_path = 'models/ocr(PN).pth'
model.load_state_dict(torch.load(model_path))
example_input = torch.rand(1, 1, 32, 100)
torch.onnx.export(model, example_input, 'models/ocr(PN).onnx', opset_version=16)
onnx.checker.check_model("models/ocr(PN).onnx")
