import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode= mode
        if self.mode=="train":
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_A_path= self.files_A[index % len(self.files_A)]
        item_A_path= item_A_path.split("\\")[-1]
        if self.mode=="train":
            if self.unaligned:
                item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
                item_B_path = self.files_B[random.randint(0, len(self.files_B) - 1)]
                item_B_path= item_B_path.split("\\")[-1]
            else:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
                item_B_path = self.files_B[index % len(self.files_B)]
                item_B_path= item_B_path.split("\\")[-1]
        if self.mode=="train":
            return {'A': item_A , 'Path_A': item_A_path, 'B': item_B , 'Path_B': item_B_path}
        else:
            return {'A': item_A , 'Path_A': item_A_path}

    def __len__(self):
        if self.mode=="train":
            return(max(len(self.files_A), len(self.files_B)))
        else:
            return(len(self.files_A))