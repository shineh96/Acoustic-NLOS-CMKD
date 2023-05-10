from models.Att_Unet import models
from models import criterion 
from data_loader.data_loader import EchoVisualDataset
from options.train_options import TrainOptions
import torch
import numpy as np 
from torch.utils.data import DataLoader
from util.util import compute_errors 
import torchvision.utils as vutils 
from util.mssim import MSSSIM
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True 

opt = TrainOptions().parse()
opt.device = torch.device('cuda')
n_GPU = torch.cuda.device_count() 

# dataset loader 
dataset = EchoVisualDataset(opt)
train_dataset = EchoVisualDataset(opt)

opt.mode = 'val'
val_dataset = EchoVisualDataset(opt) 
test_dataset = EchoVisualDataset(opt)
opt.mode = 'train'

train_loader = DataLoader(train_dataset, 
                          batch_size=opt.batch_size,
                          shuffle=opt.shuffle,
                          drop_last=True,
                          num_workers=opt.num_workers)

val_loader = DataLoader(val_dataset, 
                        batch_size=opt.batch_size,
                        shuffle=opt.shuffle,
                        drop_last=True,
                        num_workers=opt.num_workers)

print('number of samples: {}'.format(len(train_dataset) + len(val_dataset)))
print('number of train: {}'.format(len(train_dataset)))
print('number of val: {}'.format(len(val_dataset)))

# model build 
net_generator = models.Generator(opt)
net_generator = torch.nn.DataParallel(net_generator, device_ids=opt.gpu_ids)
net_generator.to(opt.device) 

criterion_L1 = torch.nn.L1Loss().to(opt.device)
criterion_MSSSIM = MSSSIM().to(opt.device)

optimizer_generator = torch.optim.Adam(net_generator.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))

writer = SummaryWriter('./logs/' + opt.version + '/')

best_rmse = float('inf')

for epoch in range(1, opt.epochs + 1):

    avg_generator_loss = avg_generator_L1_loss = avg_generator_val_loss = avg_generator_MSSSIM_loss = 0
    avg_generator_val_acc = avg_generator_val_MSSSIM_loss = 0

    net_generator.train()

    stream = tqdm(train_loader, position=0, leave=True) 
    
    bb_errors = []

    for i, (rgb, depth, bbox) in enumerate(stream, start=1):
        rgb_gt = rgb.to(opt.device)
        depth_gt = depth.to(opt.device)
        bbox_gt = bbox.to(opt.device)
        fake_b = net_generator(rgb_gt) 
       
        loss_generator_MSSSIM = (1 - criterion_MSSSIM(fake_b, depth_gt)) * 84
        loss_generator_L1 = criterion_L1(fake_b[depth_gt!=0], depth_gt[depth_gt!=0]) * 100
        loss_generator = loss_generator_L1



        optimizer_generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        # Write matrics
        acc = ((1-torch.abs(fake_b - depth_gt)) * 100) 
        acc = acc[depth_gt!=0].mean() 

        avg_generator_loss += loss_generator / len(train_loader) 
        avg_generator_L1_loss += loss_generator_L1 / len(train_loader) 
        avg_generator_MSSSIM_loss += loss_generator_MSSSIM / len(train_loader) 

        loss_bb = criterion.get_bbox_mseloss(fake_b, depth_gt, bbox_gt, opt.batch_size, opt.image_size)
        bb_errors.append(loss_bb.detach().cpu())
        

    bb_error = np.array(bb_errors).mean(0)

    print('Epoch: ', epoch)
    print('Loss - Generator_L1: {:.4f}, Generator_BB: {:.4f}'.format(avg_generator_L1_loss, bb_error))
    
    writer.add_scalar('train/loss_generator', avg_generator_loss, epoch)
    writer.add_scalar('train/loss_generator_L1', avg_generator_L1_loss, epoch)    
    writer.add_scalar('train/loss_generator_BB', bb_error, epoch)
    writer.add_scalar('train/loss_generator_MSSSIM', avg_generator_MSSSIM_loss, epoch)

    if(epoch % 1 == 0):

        # validation
        net_generator.eval()

        errors = []
        mean_errors = []
        bb_errors = []

        stream = tqdm(val_loader, position=0, leave=True) 
        with torch.no_grad():
            for i, (rgb, depth, bbox) in enumerate(stream, start=1):
                rgb_gt_val = rgb.to(opt.device)
                depth_gt_val = depth.to(opt.device)
                bbox_gt = bbox.to(opt.device)

                pred_val = net_generator.forward(rgb_gt_val) 

                loss_val = criterion_L1(pred_val[depth_gt_val!=0], depth_gt_val[depth_gt_val!=0])
                loss_MSSSIM_val = 1 - criterion_MSSSIM(pred_val, depth_gt_val)

                acc_val = ((1-torch.abs(pred_val-depth_gt_val))*100) 
                acc_val = acc_val[depth_gt_val!=0].mean() 

                avg_generator_val_loss += loss_val / len(val_loader) 
                avg_generator_val_acc += acc_val / len(val_loader) 
                avg_generator_val_MSSSIM_loss += loss_MSSSIM_val / len(val_loader)

                loss_bb = criterion.get_bbox_mseloss(pred_val, depth_gt_val, bbox_gt, opt.batch_size, opt.image_size)
                bb_errors.append(loss_bb.detach().cpu())

                for idx in range(pred_val.shape[0]):
                    errors.append(compute_errors(depth_gt_val[idx].cpu().numpy(), pred_val[idx].cpu().numpy()))

        bb_error = np.array(bb_errors).mean(0)
        mean_errors = np.array(errors).mean(0)

        print("Val - Loss: {:.4f}, Acc: {:.3f}, BB: {:.4f}".format(avg_generator_val_loss, avg_generator_val_acc, bb_error))

        if bb_error < best_rmse:
            best_rmse =  bb_error
            print('Saving the best model (epoch %d) with validation BB_RMSE %.5f' % (epoch, bb_error))
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_generator.state_dict(),
                'optimizer_state_dict': optimizer_generator.state_dict(),
            }, './logs/' + opt.version + '/' + "best_generator" + ".pt")
   
            images = vutils.make_grid(pred_val, nrow=4, normalize=True, scale_each=True)
            writer.add_image('Val/Pred', images, epoch)
            
            images = vutils.make_grid(depth_gt_val, nrow=4, normalize=True, scale_each=True)
            writer.add_image('Val/True', images, epoch)
            
        elif epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_generator.state_dict(),
                'optimizer_state_dict': optimizer_generator.state_dict(),
                }, './logs/' + opt.version + '/' + "generator_" + ".pt")

        writer.add_scalar('val/abs_rel', mean_errors[0], epoch)
        writer.add_scalar('val/rmse', mean_errors[1], epoch)
        writer.add_scalar('val/delta1', mean_errors[2], epoch)
        writer.add_scalar('val/delta2', mean_errors[3], epoch)
        writer.add_scalar('val/delta3', mean_errors[4], epoch)

        writer.add_scalar('val/loss', avg_generator_val_loss, epoch)
        writer.add_scalar('val/acc', avg_generator_val_acc, epoch)
        writer.add_scalar('val/loss_bb', bb_error, epoch)        
        writer.add_scalar('val/loss_MSSSIM', avg_generator_val_MSSSIM_loss, epoch)
