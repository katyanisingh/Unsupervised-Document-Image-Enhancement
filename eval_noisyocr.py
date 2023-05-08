try:
    from PIL import Image, ImageOps
except ImportError:
    import Image
from matplotlib.pyplot import text
import os
import cv2
import numpy as np
import easyocr
from paddleocr import PaddleOCR
import Levenshtein
import tesserocr


count=0
word_level=0
cer_level=0
#Place the ground truth text files in the same folder as the output images
path_to_file ="./noisyocr/" 

# Uncomment for PaddleOCR
#reader= PaddleOCR(use_angle_cls=True, lang='en')
def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """

    #Tesseract
    img= Image.open(filename)
    api_single_line = tesserocr.PyTessBaseAPI(lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK, path="C:/Program Files/Tesseract-OCR/tessdata", oem=tesserocr.OEM.LSTM_ONLY)
    api_single_line.SetImage(img)
    label = api_single_line.GetUTF8Text().strip()
    return label

    # easyocr

    # reader = easyocr.Reader(['en'])
    # img= Image.open(filename)
    # img = ImageOps.grayscale(img)
    # img = np.asarray(img)
    # text= reader.readtext(img, detail=0, paragraph=True)
    # return text

    #PaddleOCR 

    # img= Image.open(filename)
    # img = ImageOps.grayscale(img)
    # img = np.asarray(img)
    # text=[]
    # output= reader.ocr(img)
    # for i in range (len(output[0])):
    #     text.append(output[0][i][1][0])
    # return text

words_total=0

for file_name in os.listdir(path_to_file):
    if file_name.split(".")[-1].lower() in {"tiff"}:
        file_name_text= file_name[:-5]+".txt"
        file_name_text2= file_name[:-5]+"pred"+".txt"
        save_path = path_to_file
        name = os.path.join(save_path,file_name_text)
        name2= os.path.join(save_path,file_name_text2)
        pred=ocr_core(path_to_file+file_name)
        f = open(name2, "w", encoding="utf8")

        #UNCOMMENT for EasyOCR and PaddleOCR
        #final_str= " ".join(str(pred[i]) for i in range(len(pred)))
        #f.write(final_str)

        #COMMENT OUT NEXT LINE FOR EasyOCR and PaddleOCR
        f.write(pred)

        f.close()

        with open(name, 'r', encoding="utf8") as f:
            passage_1 = f.read().replace('\n', ' ')
        with open(name2, 'r', encoding="utf8") as f:
            passage_2 =f.read().replace('\n', ' ')

        words_total+=len(passage_1.split(" ")[:-1])
        word_level+=Levenshtein.distance(passage_2, passage_1)
        count+=1


print("Levenshtein dist:", word_level/count)
print("Total words:",words_total)

    
            
