from vslite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import onnxruntime as rt


# model_path = 'models/plate_detection_mb2-ssd-lite-Epoch-27-Loss-1.116383162962003.pth' #best model yet
image = '/home/user/ALPR/inference/ssd/plate3.jpg'


sess = rt.InferenceSession(
    "models/car_detection.onnx", providers=['CPUExecutionProvider'])
predictor_onnx = create_mobilenetv2_ssd_lite_predictor(sess, candidate_size=200)
image = cv2.imread(image)
ill_image = image.copy()
# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
# while True:
image = cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
boxes, labels, probs = predictor_onnx.predict(image, 1, 0.5)
for i in range(boxes.size(0)):
    box = boxes[i, :]
    x1,y1,x2,y2 = [int(i) for i in box]
    img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    cv2.rectangle(ill_image,(x1,y1),(x2,y2),(255,0,255))

        
cv2.imshow("Nkar ONNX",ill_image)
cv2.waitKey(1)