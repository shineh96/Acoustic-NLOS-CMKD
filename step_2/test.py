from models.NS_3DCNN import models 
from data_loader.data_loader import EchoVisualDataset
from options.test_options import TestOptions
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils 
from util.util import compute_errors 
import numpy as np 
from tqdm import tqdm 
import os 
from util.mssim import MSSSIM
from util.mssim import SSIM
from models import criterion 

opt = TestOptions().parse()
opt.device = torch.device('cuda')
n_GPU = torch.cuda.device_count() 

dataset = EchoVisualDataset(opt) 
test_loader = DataLoader(dataset, 
                        batch_size=opt.batch_size,
                        drop_last=True)
model = opt.version
test_path = opt.test_path

generator = models.Generator(opt)
checkpoint = torch.load('./logs/' + model + '/best_generator.pt', map_location=opt.device)
checkpoint['model_state_dict'] = {key.replace('module.', ''):value for key, value in checkpoint['model_state_dict'].items()}

generator.load_state_dict(checkpoint['model_state_dict']) 
generator = torch.nn.DataParallel(generator, device_ids=opt.gpu_ids) 
generator.to(opt.device)
generator.eval()


# Init loggers 
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) 


index = 0
ssim_loss = SSIM()
ms_ssim_loss = MSSSIM()

errors = []
ssim_errors = []
ms_ssim_errors = []
psnrs = []
mean_errors = []

abs_rel = []
a1 = []
a2 = []
a3 = []
rmse = []
mse = []
bb_psnr = []

stream = tqdm(test_loader, position=0, leave=True)

with torch.no_grad():
    start.record() 
    for i, (audio, rgb, depth, bbox) in enumerate(stream, start=1): 
        audio_input = audio.to(opt.device)
        depth_gt = depth.to(opt.device) 
        bbox_gt = bbox.to(opt.device)

        pred, e1, e2, e3, e4 = generator.forward(audio_input)
        #pred = generator.forward(audio_input)
        
        for idx in range(pred.shape[0]):
            errors.append(compute_errors(depth_gt[idx].cpu().numpy(), pred[idx].cpu().numpy())) 


        ssim_errors.append(ssim_loss(depth_gt.cpu(), pred.cpu()))
        ms_ssim_errors.append(1 - ms_ssim_loss(depth_gt, pred))
        psnrs.append(criterion.psnr(pred.cpu().numpy(), depth_gt.cpu().numpy()))
        
        abs_rel_, a1_, a2_, a3_, rmse_, mse_, psnr_ = criterion.get_bbox_loss_1(pred.cpu().numpy(), depth_gt.cpu().numpy(), bbox_gt, opt.batch_size, opt.image_size)
        abs_rel.append(abs_rel_)
        a1.append(a1_)
        a2.append(a2_)
        a3.append(a3_)        
        rmse.append(rmse_)
        mse.append(mse_)
        bb_psnr.append(psnr_)

    end.record() 

torch.cuda.synchronize() 

print('---------------------')
print(start.elapsed_time(end)) 


ssim_errors = list(ssim_error.cpu() for ssim_error in ssim_errors)
ssim_errors = np.array(ssim_errors).mean(0)
ms_ssim_errors = list(ms_ssim_error.cpu() for ms_ssim_error in ms_ssim_errors)
ms_ssim_errors = np.array(ms_ssim_errors).mean(0)
psnrs = np.array(psnrs).mean(0)

mean_errors = np.array(errors).mean(0)

abs_rel = np.array(abs_rel).mean(0)
a1 = np.array(a1).mean(0)
a2 = np.array(a2).mean(0)
a3 = np.array(a3).mean(0)
rmse = np.array(rmse).mean(0)
mse = np.array(mse).mean(0)
bb_psnr = np.array(bb_psnr).mean(0)

print(model)
print(test_path)

print('ABS_REL: {:.3f}'.format(mean_errors[0]))
print('RMSE: {:.3f}'.format(mean_errors[1])) 
print('DELTA1: {:.3f}'.format(mean_errors[2]))
print('DELTA2: {:.3f}'.format(mean_errors[3]))
print('DELTA3: {:.3f}'.format(mean_errors[4]))
print()
print('SSIM: {:.3f}'.format(ssim_errors))
print('MS_SSIM: {:.3f}'.format(ms_ssim_errors))
print('PSNR: {:.3f}'.format(psnrs))
print()
print('abs_rel: {:.3f}'.format(abs_rel))
print('bb_a1: {:.3f}'.format(a1))
print('bb_a2: {:.3f}'.format(a2))
print('bb_a3: {:.3f}'.format(a3))
print('bb_rmse: {:.3f}'.format(rmse))
print('bb_mse: {:.3f}'.format(mse))
print('bb_psnr: {:.3f}'.format(bb_psnr))