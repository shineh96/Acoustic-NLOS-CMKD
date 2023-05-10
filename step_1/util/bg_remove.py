#from data_loader import EchoVisualDataset
#from torch.utils.data import DataLoader
#import librosa
import numpy as np
import os
#from options.train_options import TrainOptions
from rembg.bg import remove 
import io
import cv2 
from tqdm import tqdm 
from PIL import Image

def parse_all_data(root_path):
    list_of_directories = []

    for base, dirs, files in os.walk(root_path):
        for directories in dirs:
            list_of_directories.append(directories)
    
    return list_of_directories

rename = False
bg_remove = True   

root_path = '/home/sewo/Documents/Data/aligned/F2'
input = 'gt_depth_color_image.png'
output = 'gt_depth_color_bg_remove_image.png'
output2 = 'gt_depth_image_jet_bg_remove_black_mask.png'

list_of_directories = parse_all_data(root_path)

def remove_bg(input, output):
    f = np.fromfile(input) 
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert('RGB')
    img.save(output)

def black_mask(input, output):
    image = cv2.imread(input)
    lower_black = np.array([0, 0, 0], dtype='uint16')
    upper_black = np.array([70, 70, 70], dtype='uint16')
    mask = cv2.inRange(image, lower_black, upper_black)
    cv2.imwrite(output, mask)

if bg_remove:
    for i in tqdm(list_of_directories):
        input_path = root_path + '/' + i + '/' + input 
        output_path = root_path + '/' + i + '/' + output 
        #output_path_2 = root_path + '/' + i + '/' + output2 
        remove_bg(input_path, output_path)
        #black_mask(output_path, output_path_2)

if rename:
    num = 1
    for i in list_of_directories:
        os.rename(root_path + '/' + i, 
                  root_path + '/' + 'M' + str(num).zfill(5))
        num += 1
