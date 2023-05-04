from vslite import create_mobilenetv2_ssd_lite_predictor
import cv2
import onnxruntime as rt
from configs.general import general_configs

image = '/home/user/ALPR/inference/ssd/plate3.jpg'


sess = rt.InferenceSession(
    "ALPR_python-master/models/car_detection.onnx", providers=['CPUExecutionProvider' if general_configs['device'].type == 'cpu' else 'CUDAExecutionProvider'])
predictor_onnx = create_mobilenetv2_ssd_lite_predictor(sess, candidate_size=200, device=general_configs['device'])

image = cv2.imread(image)
ill_image = image.copy()

image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
boxes, labels, probs = predictor_onnx.predict(image, 1, 0.5)
for i in range(boxes.size(0)):
    box = boxes[i, :]
    x1, y1, x2, y2 = [int(i) for i in box]
    img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    cv2.rectangle(ill_image, (x1, y1), (x2, y2), (255, 0, 255))
        
cv2.imshow("Nkar ONNX",ill_image)
cv2.waitKey(1)
