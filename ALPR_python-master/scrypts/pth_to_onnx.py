import torch
import onnx
from ocr.model import Model

model = torch.nn.DataParallel(Model())
# model = Model()
model_path = 'ALPR_python-master/models/ocr(PN).pth'
model.load_state_dict(torch.load(model_path))
example_input = torch.rand(1, 1, 32, 100)
torch.onnx.export(model, example_input, 'ALPR_python-master/models/ocr(PN).onnx', opset_version=16)
onnx.checker.check_model("ALPR_python-master/models/ocr(PN).onnx")
