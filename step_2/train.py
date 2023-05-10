from models.NS_3DCNN import models
from models import criterion 
from data_loader.data_loader import EchoVisualDataset
from options.train_options import TrainOptions
import torch
import numpy as np 
from torch.utils.data import DataLoader
from util.util import compute_errors 
from util.mssim import MSSSIM
import torchvision.utils as vutils 
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
                          num_workers=opt.num_workers,
                          pin_memory=True)
                          
val_loader = DataLoader(val_dataset, 
                        batch_size=opt.batch_size,
                        shuffle=opt.shuffle,
                        drop_last=True,
                        num_workers=opt.num_workers,
                        pin_memory=True)

print('number of samples: {}'.format(len(train_dataset) + len(val_dataset)))
print('number of train: {}'.format(len(train_dataset)))
print('number of val: {}'.format(len(val_dataset)))

# teacher_model load

t_model = models.t_Generator(opt)
checkpoint = torch.load('../step_1/logs/' + 'test' +'/best_generator.pt', map_location=opt.device)
checkpoint['model_state_dict'] = {key.replace('module.', ''):value for key, value in checkpoint['model_state_dict'].items()}
t_model.load_state_dict(checkpoint['model_state_dict']) 
t_model = torch.nn.DataParallel(t_model, device_ids=opt.gpu_ids) 
t_model.to(opt.device)
t_model.eval()

# model build 
net_generator = models.Generator(opt)
#############################
# image to image pretrain
net_generator.generator.load_state_dict(checkpoint['model_state_dict'], strict=False) 
#############################
net_generator = torch.nn.DataParallel(net_generator, device_ids=opt.gpu_ids)
net_generator.to(opt.device) 

net_discriminator = models.Discriminator(opt) 
net_discriminator = torch.nn.DataParallel(net_discriminator, device_ids=opt.gpu_ids)
net_discriminator.to(opt.device) 

criterion_GAN = criterion.GANLoss().to(opt.device)
criterion_MSSSIM = MSSSIM().to(opt.device)
criterion_KD = criterion.knowledge_distillation_loss
criterion_L1 = torch.nn.L1Loss().to(opt.device)
criterion_L2 = torch.nn.MSELoss().to(opt.device)

optimizer_generator = torch.optim.Adam(net_generator.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))
optimizer_discriminator = torch.optim.Adam(net_discriminator.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))

writer = SummaryWriter('./logs/' + opt.version + '/')

best_rmse = float('inf')
best_a1 = 0.0

torch.autograd.set_detect_anomaly(True)

for epoch in range(1, opt.epochs + 1):
    
    print('Epoch: ', epoch)

    avg_generator_loss = avg_generator_gan_loss = avg_generator_L1_loss = avg_generator_KD_loss = avg_generator_val_loss = avg_generator_MSSSIM_loss =0
    avg_generator_acc = avg_generator_gan_acc = avg_generator_val_acc = avg_generator_val_MSSSIM_loss = avg_generator_val_KD = 0

    avg_discriminator_loss = avg_discriminator_val_loss = 0
    avg_discriminator_real_acc = avg_discriminator_fake_acc = 0
 
    avg_generator_KD1_loss = avg_generator_KD4_loss = 0
 
    net_discriminator.train()
    net_generator.train()
    t_model.eval()

    stream = tqdm(train_loader, position=0, leave=True) 
    
    bb_errors = []

    for i, (audio, rgb, depth, bbox) in enumerate(stream, start=1):
   
        audio_input = audio.to(opt.device)
        rgb_gt = rgb.to(opt.device)
        depth_gt = depth.to(opt.device)
        bbox_gt = bbox.to(opt.device)
        
        fake_b, e1, e2, e3, e4 = net_generator(audio_input) 
        fake_t, t1, t2, t3, t4 = t_model.forward(rgb_gt)

        # (1) Update Discriminator network
        # train with fake
        pred_fake = net_discriminator(fake_b.detach())
        loss_discriminator_fake = criterion_GAN(pred_fake, False)

        label = torch.full((pred_fake.size(0), pred_fake.size(-1), pred_fake.size(-1)), 0, device=opt.device)
        acc_discriminator_fake = ((1-torch.abs(pred_fake - label))*100).mean()

        # train with real
        pred_real = net_discriminator(depth_gt) 
        loss_discriminator_real = criterion_GAN(pred_real, True) 

        label.fill_(1) 
        acc_discriminator_real = ((1-torch.abs(pred_real - label))*100).mean()

        # Combined D loss    
        loss_discriminator = 0.5*(loss_discriminator_fake + loss_discriminator_real)

        optimizer_discriminator.zero_grad()  
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # (2) Update Generator network

        # Firsty try to fake the discriminator
        pred_fake = net_discriminator(fake_b)
        
        ###############
        loss_generator_gan = criterion_GAN(pred_fake, True)
        ###############
        
        label.fill_(1)
        acc_generator_gan = ((1-torch.abs(pred_fake - label))*100).mean()
            
        loss_kd1 = criterion_KD(e1,t1,T=opt.kd_temperature) * 0.01
        loss_kd2 = criterion_KD(e2,t2,T=opt.kd_temperature) * 0.01
        loss_kd3 = criterion_KD(e3,t3,T=opt.kd_temperature) * 0.01
        loss_kd4 = criterion_KD(e4,t4,T=opt.kd_temperature) * 0.01
        loss_kd = loss_kd1 + loss_kd2 + loss_kd3 + loss_kd4

        loss_generator_MSSSIM = (1 - criterion_MSSSIM(fake_b, depth_gt))
        loss_generator_L1 = criterion_L1(fake_b[depth_gt!=0], depth_gt[depth_gt!=0]) * 100
        loss_generator = loss_generator_gan + loss_generator_L1 + loss_kd

        optimizer_generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        # (3) Write matrics
        acc = ((1-torch.abs(fake_b - depth_gt)) * 100) 
        acc = acc[depth_gt!=0].mean() 

        avg_generator_KD_loss += loss_kd / len(train_loader) 
        avg_generator_KD1_loss += loss_kd1 / len(train_loader) 
        avg_generator_KD4_loss += loss_kd4 / len(train_loader) 
        avg_generator_loss += loss_generator / len(train_loader) 
        avg_generator_MSSSIM_loss += loss_generator_MSSSIM / len(train_loader) 
        avg_generator_gan_loss += loss_generator_gan / len(train_loader)
        avg_generator_L1_loss += loss_generator_L1 / len(train_loader) 
        avg_discriminator_loss += loss_discriminator / len(train_loader) 
        avg_generator_acc += acc / len(train_loader) 
        avg_generator_gan_acc += acc_generator_gan / len(train_loader) 
        avg_discriminator_fake_acc += acc_discriminator_fake / len(train_loader)
        avg_discriminator_real_acc += acc_discriminator_real / len(train_loader)

        loss_bb = criterion.get_bbox_mseloss(fake_b, depth_gt, bbox_gt, opt.batch_size, opt.image_size)
        bb_errors.append(loss_bb.detach().cpu())

    bb_error = np.array(bb_errors).mean(0)

    print('Loss - Discriminator: {:.4f}, Generator: {:.4f}, Generator_gan: {:.4f}, Generator_L1: {:.4f}, Generator_BB: {:.4f}, Generator_MSSSIM: {:.4f}'.format(avg_discriminator_loss, avg_generator_loss, avg_generator_gan_loss, avg_generator_L1_loss, bb_error, avg_generator_MSSSIM_loss))
    print('KDLoss - KD: {:.4f}, KD1: {:.4f}, KD4: {:.4f}'.format(avg_generator_KD_loss, avg_generator_KD1_loss, avg_generator_KD4_loss))
    print('Acc - Generator_L1: {:.3f}, Generator_gan: {:.3f}, Discriminator_real: {:.3f}, Discriminator_fake: {:.3f}'.format(avg_generator_acc, avg_generator_gan_acc, avg_discriminator_real_acc, avg_discriminator_fake_acc))
    
    writer.add_scalar('train/loss_discriminator', avg_discriminator_loss, epoch)
    writer.add_scalar('train/loss_generator', avg_generator_loss, epoch)
    writer.add_scalar('train/loss_generator_gan', avg_generator_gan_loss, epoch)
    writer.add_scalar('train/loss_generator_L1', avg_generator_L1_loss, epoch)
    writer.add_scalar('train/loss_generator_KD', avg_generator_KD_loss, epoch)    
    writer.add_scalar('train/loss_generator_BB', bb_error, epoch)
    writer.add_scalar('train/loss_generator_MSSSIM', avg_generator_MSSSIM_loss, epoch)

    if(epoch % 1 == 0):

        # validation
        net_generator.eval()
        t_model.eval()

        errors = []
        mean_errors = []
        bb_errors = []
        bb_a1_ = []

        stream = tqdm(val_loader, position=0, leave=True) 
        with torch.no_grad():
            for i, (audio, rgb, depth, bbox) in enumerate(stream, start=1):
                audio_input = audio.to(opt.device)
                rgb_gt_val = rgb.to(opt.device)
                depth_gt_val = depth.to(opt.device)
                bbox_gt = bbox.to(opt.device)

                pred_val, e1, e2, e3, e4 = net_generator.forward(audio_input) 
                fake_t, t1, t2, t3, t4 = t_model.forward(rgb_gt_val)

                loss_val_kd1 = criterion_KD(e1,t1) * 0.01
                loss_val_kd2 = criterion_KD(e2,t2) * 0.01
                loss_val_kd3 = criterion_KD(e3,t3) * 0.01
                loss_val_kd4 = criterion_KD(e4,t4) * 0.01
                loss_val_kd = loss_val_kd1 + loss_val_kd2 + loss_val_kd3 + loss_val_kd4

                loss_MSSSIM_val = 1 - criterion_MSSSIM(pred_val, depth_gt_val)
                loss_val = criterion_L1(pred_val[depth_gt_val!=0], depth_gt_val[depth_gt_val!=0])

                acc_val = ((1-torch.abs(pred_val-depth_gt_val))*100) 
                acc_val = acc_val[depth_gt_val!=0].mean() 

                avg_generator_val_loss += loss_val / len(val_loader) 
                avg_generator_val_KD += loss_val_kd / len(val_loader) 
                avg_generator_val_MSSSIM_loss += loss_MSSSIM_val / len(val_loader)
                avg_generator_val_acc += acc_val / len(val_loader) 

                abs_rel_, a1_, a2_, a3_, rmse_, mse_, psnr_ = criterion.get_bbox_loss(pred_val.cpu().numpy(), depth_gt_val.cpu().numpy(), bbox_gt, opt.batch_size, opt.image_size)
                bb_a1_.append(a1_)
                bb_errors.append(mse_)

                for idx in range(pred_val.shape[0]):
                    errors.append(compute_errors(depth_gt_val[idx].cpu().numpy(), pred_val[idx].cpu().numpy()))
       
        bb_a1 = np.array(bb_a1_).mean(0)
        bb_error = np.array(bb_errors).mean(0)
        mean_errors = np.array(errors).mean(0)

        print("Val - Loss: {:.4f}, Acc: {:.3f}, BB: {:.4f}, MSSSIM: {:.4f}, KD: {:.4f}".format(avg_generator_val_loss, avg_generator_val_acc, bb_error, avg_generator_val_MSSSIM_loss, avg_generator_val_KD))

        if bb_error < best_rmse:
            best_rmse =  bb_error
            print('Saving the best model (epoch %d) with validation bb_error %.5f' % (epoch, bb_error))
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_generator.state_dict(),
                'optimizer_state_dict': optimizer_generator.state_dict(),
            }, './logs/' + opt.version + '/' + "best_generator" + ".pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_discriminator.state_dict(),
                'optimizer_state_dict': optimizer_discriminator.state_dict(),
                }, './logs/' + opt.version + '/' + "best_discriminator" + ".pt")
            
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
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_discriminator.state_dict(),
                'optimizer_state_dict': optimizer_discriminator.state_dict(),
                }, './logs/' + opt.version + '/' + "discriminator_" + ".pt")

        writer.add_scalar('val/abs_rel', mean_errors[0], epoch)
        writer.add_scalar('val/rmse', mean_errors[1], epoch)
        writer.add_scalar('val/delta1', mean_errors[2], epoch)
        writer.add_scalar('val/delta2', mean_errors[3], epoch)
        writer.add_scalar('val/delta3', mean_errors[4], epoch)

        writer.add_scalar('val/loss', avg_generator_val_loss, epoch)
        writer.add_scalar('val/acc', avg_generator_val_acc, epoch)
        writer.add_scalar('val/bb_delta1', bb_a1, epoch)
        writer.add_scalar('val/loss_bb', bb_error, epoch)
        writer.add_scalar('val/loss_MSSSIM', avg_generator_val_MSSSIM_loss, epoch)
        writer.add_scalar('val/loss_KD', avg_generator_val_KD, epoch)