import torch
import argparse
import torchvision.transforms as transforms
from patch_dataset import PatchDataset
from utils_eval import compare_labels, get_text_stack, get_ocr_helper
import properties as properties
from PIL import Image


class EvalPrep():

    def __init__(self, args):
        self.batch_size = 1
        self.show_txt = args.show_txt
        self.ocr_name = args.ocr  
        self.test_set = properties.patch_dataset_test
        self.input_size = properties.input_size
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.ocr = get_ocr_helper(self.ocr_name, is_eval=True)
        self.dataset = PatchDataset(self.test_set, pad=True)

    def eval_patch(self):
        print("Eval with ", self.ocr_name)
        ori_lbl_crt_count = 0
        ori_lbl_cer = 0
        lbl_count = 0
        

        for image, labels_dict in self.dataset:
            
            text_crops, labels = get_text_stack(
                image.detach(), labels_dict, self.input_size)
            lbl_count += len(labels)
            ocr_labels = self.ocr.get_labels(text_crops)
            
            ori_crt_count, ori_cer= compare_labels(
                ocr_labels, labels)
            ori_lbl_crt_count += ori_crt_count
            ori_lbl_cer += ori_cer
            ori_cer = round(ori_cer/len(labels), 2)


        print('Correct Word-accuracy from images: {:d}/{:d} ({:.5f})'.format(ori_lbl_crt_count, lbl_count, ori_lbl_crt_count/lbl_count))
        print('Average CER from images: ({:.5f})'.format(ori_lbl_cer/lbl_count))

    def eval(self):
        self.eval_patch()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--show_txt', action='store_true',
                        help='prints predictions and groud truth')
 
    parser.add_argument('--ocr', default="Tesseract",
                        help="performs training lebels from given OCR [Tesseract,EasyOCR, PaddleOCR]")
    parser.add_argument("--prep_model_name",
                        default='prep_tesseract_pos', help='Prep model name')
    args = parser.parse_args()
    print(args)
    evaluator = EvalPrep(args)
    evaluator.eval()
