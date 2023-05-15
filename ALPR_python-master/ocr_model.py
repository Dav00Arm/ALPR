import torch.nn as nn
from ocr_TPS import TPS_SpatialTransformerNetwork
from ocr_feature_extractor import ResNet_FeatureExtractor
from ocr_BLSTM import BidirectionalLSTM
from ocr_attention import Attention
import torch


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.stages = {'Trans': 'TPS', 'Feat': 'ResNet',
                       'Seq': 'BiLSTM', 'Pred': "Attn"}

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, I_size=(32,100), I_r_size=(32,100), I_channel_num=1)

        """ FeatureExtraction """

        # elif opt.FeatureExtraction == 'ResNet':
        self.FeatureExtraction = ResNet_FeatureExtractor(1,512)
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((512,1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256

        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, 256, 96)

    def forward(self, input):
        """ Transformation stage """

        text = torch.zeros(1, 12, dtype=torch.float64)

        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, batch_max_length=11)

        return prediction
