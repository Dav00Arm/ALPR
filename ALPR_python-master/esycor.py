"""CLEANED"""
import torch
import torch.utils.data
import torch.nn.functional as F
from cfour import *

def test_ocr(lines):
    plate = ''
    conf = 0
    for line in lines:
        image_tensors = AlignCollate_demo(line)
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

        preds = plate_recognition_model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            # if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1].cpu().detach().numpy()
        plate += pred 
        conf += confidence_score 
            # print(f'{1}\t{pred:25s}\t{confidence_score:0.4f}')
    conf = conf/len(lines) if len(lines) > 0 else 0
    return plate, int(conf*100)