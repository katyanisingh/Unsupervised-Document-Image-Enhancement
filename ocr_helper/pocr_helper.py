import torch
import numpy as np
from torchvision.transforms import ToPILImage
from paddleocr import PaddleOCR
import properties as properties
import utils_eval


class PocrHelper():
    def __init__(self, empty_char=properties.empty_char, is_eval=False):
        self.empty_char = empty_char
        self.is_eval = is_eval
        self.reader= PaddleOCR(use_angle_cls=True, lang='en')


    def get_labels(self, imgs):
        labels = []
        for i in range(imgs.shape[0]):
            img = ToPILImage()(imgs[i])
            img = np.asarray(img)
            output = self.reader.ocr(img, det=False,cls=True)
            label=output[0][0][0]
            if label=='':
                label = self.empty_char
            if self.is_eval:
                labels.append(label)
                continue
            label = utils_eval.get_ununicode(label)
            if len(label) > properties.max_char_len:
                label = self.empty_char
            labels.append(label)
        return labels

    def get_string(self, img):
        img = ToPILImage()(img)
        img = np.asarray(img)
        output = self.reader.ocr(img, det=False,cls=True)
        string = output[0][0][0]
        for i in range(len(string)):
            string[i] = utils_eval.get_ununicode(string[i])
        return string
