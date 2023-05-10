import os 
import argparse 
from util import util 

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # dataset options  
        self.parser.add_argument('--train_path', type=str, default='./Data/aligned', help='train path to the files')
        self.parser.add_argument('--val_path', type=str, default='./Data/aligned', help='validation path to the files')
        self.parser.add_argument('--test_path', type=str, default='./Data/aligned', help='test path to the files')
        self.parser.add_argument('--split', type=str, default='True', help='[True, False] split channels or not')
        self.parser.add_argument('--n_fft', type=int, default=512, help='length of the windowed signal after padding with zeros')
        self.parser.add_argument('--win_length', type=int, default=64, help='Each frame of audio is windowed by window of length')
        self.parser.add_argument('--batch_size', type=int, default=64, help='how many samples per batch to load')
        self.parser.add_argument('--shuffle', type=bool, default=True, help='data reshuffled at every epoch')
        self.parser.add_argument('--image_size', type=int, default=128, help='input & output image size' )
        self.parser.add_argument('--grayscale', type=str, default=False, help='[True, False] grayscale image or rgb image')
        self.parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--audio_sampling_rate', type=int, default=48000, help='audio sampling rate')
        self.parser.add_argument('--checkpoints_dir', type=str, default='logs/', help='path to save checkpoints')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 0,1,2 0,2. use -1 for CPU')
        self.parser.add_argument('--version', type=str, default='1', help='version of model')  
        self.parser.add_argument('--image_augmentation', type=bool, default=False, help='whether to augmentation the image data')
        self.parser.add_argument('--kd_temperature', type=int, default=9, help='knowledge distillation temperature')
        
        self.initialized = True 
    
    def parse(self):
        if not self.initialized:
            self.initialize() 
        
        self.opt = self.parser.parse_args() 
        self.opt.mode = self.mode 
        self.opt.isTrain = self.isTrain 

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if(self.isTrain):
            args = vars(self.opt)
            print('---------- Options ----------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('------------ End ------------')

            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.version)
            self.opt.expr_dir = expr_dir
            util.mkdirs(expr_dir) 
            file_name = os.path.join(self.opt.expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('---------- Options ----------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('------------ End ------------\n')
            
        return self.opt