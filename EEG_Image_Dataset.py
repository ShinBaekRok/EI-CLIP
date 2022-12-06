
from torchvision import transforms
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image


data_list = torch.load('/users/sbr/data/eeg_55_95_std.pth')['dataset']
label_list = torch.load('/users/sbr/data/eeg_55_95_std.pth')['labels']
image_list = torch.load('/users/sbr/data/eeg_55_95_std.pth')['images']

label_path = list(map(lambda x: x.split('/')[-1], glob.glob('/users/sbr/data/Image_data/' + '*')))
all_image_path = []
for label in label_path:
     all_image_path.extend(glob.glob('/users/sbr/data/Image_data/' + label + '/*'))
        
all_image_list = list(map(lambda x: x.split('.')[0].split('/')[-1], all_image_path))

image_path = []
for image in image_list:
    label = image.split('_')[0]
    image_path.append('/users/sbr/data/Image_data/' + label + '/' + image + '.JPEG')
    
f =  open('/users/sbr/data/LOC_synset_mapping.txt', 'r')
label_txt = []
for line in f:
    label_txt.append(line[:9])
f.close()
label_txt = np.array(label_txt)

label_for_train = {}
for label in label_list:
    label_for_train[label] = np.where(label_txt == label)[0][0]


class EEG_Image_Dataset:
    
    # Constructor
    def __init__(self, eeg_signals_path, image_path, split): #Type of image path = list
        loaded = torch.load(eeg_signals_path)
        
        self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.image_labels = loaded["images"]
        
        self.size = len(self.data)
        
        self.image_path = '/users/sbr/data/Image_data/'
        
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomHorizontalFlip(p = 1),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)])
        
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[20:460,:]
        eeg = eeg.t()
        eeg = eeg.view(1,128,440)
        
        image_label = self.data[i]["image"]
        image_label = self.image_labels[image_label]
        class_label = self.data[i]['label']
        class_label = self.labels[class_label]
        
        image = Image.open(self.image_path + class_label + '/' + image_label + '.JPEG').convert('RGB')
        image = self.transform(image)
        
        label = self.data[i]["label"]
        
        return eeg, image, label, image_label

class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, image, label, image_label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, image, label, image_label

