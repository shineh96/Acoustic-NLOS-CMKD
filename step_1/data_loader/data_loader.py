import numpy as np
from torchvision import transforms
from PIL import Image 
import librosa 
import os 
import torch 
from torch.utils.data.dataset import Dataset

def parse_all_data(root_path):

    list_of_directories = []

    for base, dirs, files in os.walk(root_path):
        for directories in dirs:
            list_of_directories.append(directories)
        
    return list_of_directories

def to_tensor(sample, grayscale):
    if (grayscale == 'True'):
        image = sample.convert('L')
        image = np.asarray(image) 
        image = np.expand_dims(image, axis=0) 
        image = (image * 1/255) 
    else:
        image = np.asarray(sample)
        image = (image * 1/255).transpose((2, 0, 1)) 
    image = torch.FloatTensor(image) 

    return image

class EchoVisualDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        if(opt.mode == 'train'): self.root_path = opt.train_path
        elif(opt.mode == 'val'): self.root_path = opt.val_path 
        else: self.root_path = opt.test_path

        self.split = opt.split 
        self.n_fft = opt.n_fft 
        self.win_length = opt.win_length
        self.image_size = opt.image_size
        self.grayscale = opt.grayscale    
        self.list_of_directories = parse_all_data(self.root_path)#os.listdir(self.root_path) 

    def __len__(self):
        return len(self.list_of_directories)

    def __getitem__(self, idx):
        bbox = np.load(os.path.join(self.root_path, self.list_of_directories[idx], 'bbox.npy'))

        rgb = Image.open(os.path.join(self.root_path, self.list_of_directories[idx], 'gt_rgb_image.png')).resize((self.image_size, self.image_size), Image.ANTIALIAS)
        rgb = to_tensor(rgb, self.grayscale) 

        depth = Image.open(os.path.join(self.root_path, self.list_of_directories[idx], 'gt_depth_gray_image.png')).resize((self.image_size, self.image_size), Image.ANTIALIAS)
        depth = to_tensor(depth, self.grayscale) 

        return rgb, depth, bbox