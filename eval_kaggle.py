try:
    from PIL import Image
except ImportError:
    import Image
import os
import Levenshtein
import cv2
import json
import numpy as np
import easyocr
from paddleocr import PaddleOCR
import tesserocr


counter=0
count=0
cer=0
total_words=0
total_sent=0

#Place the ground truth json files in the same folder as the output images
path_to_file ="./kaggle/"

# Uncomment for PaddleOCR
#reader= PaddleOCR(use_angle_cls=True, lang='en') 
def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """

    #Tesseract
    img= Image.open(filename)
    api_single_line = tesserocr.PyTessBaseAPI(lang='eng', psm=tesserocr.PSM.SINGLE_LINE, path="C:/Program Files/Tesseract-OCR/tessdata", oem=tesserocr.OEM.LSTM_ONLY)
    api_single_line.SetImage(img)
    label = api_single_line.GetUTF8Text().strip()

    return label

    # easyocr 

    # reader = easyocr.Reader(['en'])
    # img= Image.open(filename)
    # img = np.asarray(img)
    # text= reader.readtext(img, detail=0,paragraph=True)
    # return text

    # PaddleOCR

    # img= Image.open(filename)
    # img = np.asarray(img)
    # text=[]
    # output= reader.ocr(img)
    # for i in range (len(output[0])):
    #     text.append(output[0][i][1][0])
    # return text

for file_name in os.listdir(path_to_file):
    if file_name.split(".")[-1].lower() in {"png","jpg"}:
        file_name_text= file_name[:-4]+".json"
        save_path = path_to_file
        name = os.path.join(save_path,file_name_text)
        data={}
        with open(name, 'r') as f:
            counter=0
            img = cv2.imread(path_to_file+file_name)
            data = json.loads(f.read())
            for box in data:
                label_list=[]
                pred_list=[]
                counter+=1
                label = box["label"]
                x_min = int(box['x1'])
                y_min = int(box['y1'])
                x_max = int(box['x2'])
                y_max = int(box['y3'])

                text_crop = img[y_min:y_max, x_min:x_max]
                cv2.imwrite("./temp/"+str(counter)+".png",text_crop)
                pred = ocr_core("./temp/"+str(counter)+".png")
                # uncomment next line for EasyOCR and PaddleOCR
                #pred= " ".join((str(pred[i]) for i in range(len(pred)))) 
                label_list=list(filter(None, label.split(' ')))
                pred_list= list(filter(None, pred.split(' ')))

                if len(label_list)>len(pred_list):
                    for emp in range(len(label_list)-len(pred_list)):
                        pred_list.append(" ")

                for i in range(len(label_list)):
                    if label_list[i] == pred_list[i]:
                        count+=1
                    distance = Levenshtein.distance(label_list[i], pred_list[i])
                    cer+=distance/len(label_list[i])
                total_words+=len(label_list)


print("Word accuracy",count/total_words)
print("CER", cer/total_words)

