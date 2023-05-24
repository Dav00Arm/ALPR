import onnxruntime as ort
from models_decrypt import decrypt_file, load_graph


_key = "ThisistheonlyKeyThatwillwork123@%^*&((??__+(I_GAS"
save_path = "PROJECT_Car_Detection_Model_SOFTWARE_AI/models/yolov5n.onnx"
model_weights = "PROJECT_Car_Detection_Model_SOFTWARE_AI/models/yolov5n.onnx"
decrypted_weights = decrypt_file(load_graph(model_weights), _key)

with open(save_path, 'wb') as file:
    file.write(decrypted_weights)
