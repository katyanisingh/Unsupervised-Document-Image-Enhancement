# Unsupervised-Document-Image-Enhancement 
This repository contains the implementation for the work "Unpaired Document Image Denoising for OCR using BiLSTM enhanced
CycleGAN".


## Setup
Create a python virtual environment and install the required packages using
```bash
pip3 install -r requirements.txt
``` 
## Datasets
All the datasets used can be downloaded from the link below. Place the "datasets" folder in the main directory. 

* [Datasets](https://drive.google.com/file/d/1c6Leomjyf6to_ElrCrPC-vM4HmvDajRT/view?usp=drive_link)


## Training 
1. Modify the training parameters and dataset name in params.json. 
2. Run the following command
```python 
  python train.py --params ./params.json --filename <name_for_saving_trained_model> --wandb_run_name <wandb_run_name> 
```

## Using Pretrained models for evaluation
1. Pretrained models are available in ./models/saved_models folder.  
2. Modify any test parameters in test.py if required. 
3. Run the following command
```python 
 python test.py --dataroot <path_to_dataset_testfolder> --filename <name_for_output_folder> --generator_A2B <pretrained_model_file>
```
 
 
## Scripts for OCR evaluation 
* eval_kaggle.py : OCR evaluation for Kaggle dataset
* eval_noisyocr.py : OCR evaluation for Noisy OCR dataset
* eval_pos_wr.py: OCR evaluation for POS and WildReceipt datasets

