import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from cfour import *
import PIL
import cv2
import onnxruntime

def test_ocr(lines):
    plate = ''
    conf = 0
    for line in lines:
        # cv2.imshow("CAP",np.array(line))
        # cv2.waitKey(0)
        image_tensors = AlignCollate_demo(line)
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor(
            [batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(
            batch_size, batch_max_length + 1).fill_(0).to(device)
        # session = onnxruntime.InferenceSession('/home/user/Desktop/models/ocr1.onnx')
        # model = torch.jit.load('models/ocr(PN).pt')

        # preds = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: np.array(image)})[0]
        preds = plate_recognition_model(image)
        # preds = plate_recognition_model(image, text_for_pred)
        preds = torch.Tensor(preds)
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for i,(pred, pred_max_prob) in enumerate(zip(preds_str, preds_max_prob)):
            # if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(
                dim=0)[-1].cpu().detach().numpy()
        plate += pred
        conf += confidence_score
        # print(f'{1}\t{pred:25s}\t{confidence_score:0.4f}')
    conf = conf/len(lines) if len(lines) > 0 else 0
    # print(plate)
    return plate, int(conf*100)


# for i in range(10):
#     image = PIL.Image.open("/home/user/deep-text-recognition-benchmark/demo_image/demo_1.png").convert('L')
#     test_ocr([image])
