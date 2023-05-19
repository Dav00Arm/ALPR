import torch
import torch.utils.data
import torch.nn.functional as F
from configs.model_configs import plate_rec_configs
from configs.general import general_configs

import numpy as np
import torch
import PIL
import cv2
import onnxruntime


def test_ocr(lines):
    plate = ''
    conf = 0
    for line in lines:

        image_tensors = plate_rec_configs['AlignCollate_demo'](line)

        batch_size = image_tensors.size(0)
        image = image_tensors.to('cpu')
        # For max length prediction
        length_for_pred = torch.IntTensor(

            [plate_rec_configs['batch_max_length']] * batch_size).to(general_configs['device'])
        text_for_pred = torch.LongTensor(
            batch_size, plate_rec_configs['batch_max_length'] + 1).fill_(0).to(general_configs['device'])

        preds = plate_rec_configs['plate_recognition_model'](image)

        preds = torch.Tensor(preds)
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = plate_rec_configs['converter'].decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for i, (pred, pred_max_prob) in enumerate(zip(preds_str, preds_max_prob)):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(
                dim=0)[-1].cpu().detach().numpy()
        plate += pred
        conf += confidence_score

    conf = conf/len(lines) if len(lines) > 0 else 0

    return plate, int(conf*100)
