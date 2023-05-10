import numpy as np
from PIL import Image 
import os 
import torch 
import torchaudio 
import torchvision.transforms as VisionT 
import torchaudio.transforms as AudioT
from torch.utils.data.dataset import Dataset

def waveform_to_stft(audio_waveform, split='True', n_fft=512):   

    spectrogram = AudioT.Spectrogram(
        n_fft=n_fft,
        #win_length=win_length,
        #hop_length=win_length//4,
        center=True,
        pad_mode='constant'
    )

    audio_waveform = np.moveaxis(audio_waveform, [0, 1, 2], [0, 2, 1])
    audio_stft = [] 
    
    for x in range(8): # audio_waveform.shape[0]
        stft_channel = []
        for y in range(audio_waveform.shape[1]): # audio_waveform.shape[1] 
            waveform = torch.tensor(audio_waveform[x][y][3600:48000-3200]) #[4500:14401]
            stft = spectrogram(waveform)[:256,:512]
            if (split=='True'): stft_channel.append(stft)
            elif (split=='False'): audio_stft.append(stft)
            else: print('split not defined')
        
        if (split=='True'): audio_stft.append(stft_channel)
    
    tensor = torch.stack([torch.stack(sub_list, dim=0) for sub_list in audio_stft], dim=0)

    return tensor



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
        self.list_of_directories = parse_all_data(self.root_path)

    def __len__(self):
        return len(self.list_of_directories)

    def __getitem__(self, idx):
        bbox = np.load(os.path.join(self.root_path, self.list_of_directories[idx], 'bbox.npy'))
              
        #audio = np.load(os.path.join(self.root_path, self.list_of_directories[idx], 'WAVE.npy'))
        #audio_stft = waveform_to_stft(audio, split=self.split, n_fft=self.n_fft, win_length=self.win_length)
        audio_stft = np.load(os.path.join(self.root_path, self.list_of_directories[idx], 'STFT_512.npy'))
        
        rgb = Image.open(os.path.join(self.root_path, self.list_of_directories[idx], 'gt_rgb_image.png')).resize((self.image_size, self.image_size), Image.ANTIALIAS)
        rgb = to_tensor(rgb, self.grayscale) 
        
        depth = Image.open(os.path.join(self.root_path, self.list_of_directories[idx], 'gt_depth_gray_image.png')).resize((self.image_size, self.image_size), Image.ANTIALIAS)
        depth = to_tensor(depth, self.grayscale) 

        return audio_stft, rgb, depth,bbox