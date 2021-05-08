import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

import cv2
from matplotlib import pyplot as plt

from avgmeter import *
from ioueval import *
import os, shutil

from losses import MTL_loss
        
def trainer_kitti(args, model, snapshot_path, parser):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
                          
    print("The length of train set is: {}".format((parser.get_train_size())))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = parser.get_train_set()#DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)#,
                             #worker_init_fn=worker_init_fn) #this gave the error Can't pickle local object 'trainer_synapse.<locals>.worker_init_fn'
    
    ########################                         
    valid_loader = parser.get_valid_set()
    device = torch.device("cuda")
    ignore_classes = [0]
    evaluator = iouEval(parser.get_n_classes(),device, ignore_classes)
    
    
    #Empty the TensorBoard directory
    dir_path = 'C:/msc_codes/proj_tansUnet/model/TU_Kitti64x1024/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs5_64x1024/log'

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
        
    # for filename in os.listdir(dir_path):
        # file_path = os.path.join(dir_path, filename)
        # try:
            # if os.path.isfile(file_path) or os.path.islink(file_path):
                # os.unlink(file_path)
            # elif os.path.isdir(file_path):
                # shutil.rmtree(file_path)
        # except Exception as e:
            # print('Failed to delete %s. Reason: %s' % (file_path, e))
    #######################
    
    
    ######################
    ######################
    if args.pretrain:
        criterion = MTL_loss(device, args.batch_size)
    ######################
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    # dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        
        iou = AverageMeter()
        
        for i_batch, batch_data in enumerate(trainloader):
        
            if args.pretrain:
                (image_batch, proj_mask, reduced_image_batch, reduced_proj_mask, \
                rot_ang_0_is_0_180_is_1_batch, rot_x_is_0_y_is_1_batch, image_batch_2, proj_mask_2, reduced_image_batch_2,\
                reduced_proj_mask_2, rot_ang_0_is_0_180_is_1_batch_2, rot_x_is_0_y_is_1_batch_2,path_seq, path_name) =  batch_data
            
                image_batch = image_batch.to(device, non_blocking=True)
                reduced_image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion
                rot_x_is_0_y_is_1_batch = rot_x_is_0_y_is_1_batch.to(device, non_blocking=True)
                rot_ang_0_is_0_180_is_1_batch = rot_ang_0_is_0_180_is_1_batch.to(device, non_blocking=True)
                
                
                image_batch_2 = image_batch_2.to(device, non_blocking=True)
                reduced_image_batch_2 = reduced_image_batch_2.to(device, non_blocking=True) # Apply distortion
                rot_x_is_0_y_is_1_batch_2 = rot_x_is_0_y_is_1_batch_2.to(device, non_blocking=True)
                rot_ang_0_is_0_180_is_1_batch_2 = rot_ang_0_is_0_180_is_1_batch_2.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                
                    rot_prd,rot_axis_prd, contrastive_prd, recon_prd, rot_w, rot_axis_w, contrastive_w, recons_w = model(reduced_image_batch)
                    rot_prd_2,rot_axis_prd_2, contrastive_prd_2, recon_prd_2, _, _, _, _                         = model(reduced_image_batch_2)
                        
                    rot_p = torch.cat([rot_prd, rot_prd_2], dim=0) 
                    rots = torch.cat([rot_ang_0_is_0_180_is_1_batch, rot_ang_0_is_0_180_is_1_batch_2], dim=0) 
                    
                    rot_axis_p = torch.cat([rot_axis_prd, rot_axis_prd_2], dim=0) 
                    rots_axis = torch.cat([rot_x_is_0_y_is_1_batch, rot_x_is_0_y_is_1_batch_2], dim=0) 
                    
                    imgs_recon = torch.cat([recon_prd, recon_prd_2], dim=0) 
                    imgs = torch.cat([image_batch, image_batch_2], dim=0) 
                    
                    loss, (loss1, loss2, loss3, loss4) = criterion(rot_p, rots, 
                                                                rot_axis_p, rots_axis,
                                                                contrastive_prd, contrastive_prd_2, 
                                                                imgs_recon, imgs, rot_w, rot_axis_w, contrastive_w, recons_w )
                    
                writer.add_scalar('info/loss_rotation', loss1, iter_num)
                writer.add_scalar('info/loss_rot_axis', loss2, iter_num)
                writer.add_scalar('info/loss_contrastive', loss3, iter_num)
                writer.add_scalar('info/loss_reconstruction', loss4, iter_num)
            else:
                (image_batch, proj_mask, label_batch, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) = batch_data
                
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda(non_blocking=True).long()
                outputs = model(image_batch)
            
                loss_ce = ce_loss(outputs, label_batch)
                # loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = loss_ce 
                
                ###########################
                
                with torch.no_grad():
                    evaluator.reset() # we do this for the training as each weights of the model differ each iteration 
                    argmax = outputs.argmax(dim=1)
                    evaluator.addBatch(argmax, label_batch)
                    jaccard, class_jaccard = evaluator.getIoU()
                iou.update(jaccard.item(), args.batch_size)
                ###########################
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss', loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # if iter_num % 20 == 0:
                # image = image_batch[1, 0:1, :, :]
                # image = (image - image.min()) / (image.max() - image.min())
                # writer.add_image('train/Image', image, iter_num)
                # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                # labs = label_batch[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/GroundTruth', labs, iter_num)
                
        if not args.pretrain:
            writer.add_scalar('train/iou', iou.avg, epoch_num)
                
                
        ##############################################
        ##############################################
        if (epoch_num + 1) % 10 == 0:
            evaluator.reset()
            iou.reset()
            
            val_losses = AverageMeter()
            
            model.eval()
            with torch.no_grad():
                
                for index, batch_data in enumerate(valid_loader):
                    if index % 100 == 0:
                        print('%d validation iter processd' % index)
                        
                    if args.pretrain:
                        (image_batch, proj_mask, reduced_image_batch, reduced_proj_mask, \
                        rot_ang_0_is_0_180_is_1_batch, rot_x_is_0_y_is_1_batch, image_batch_2, proj_mask_2, reduced_image_batch_2,\
                        reduced_proj_mask_2, rot_ang_0_is_0_180_is_1_batch_2, rot_x_is_0_y_is_1_batch_2,path_seq, path_name) =  batch_data
                    
                        image_batch = image_batch.to(device, non_blocking=True)
                        reduced_image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion
                        rot_x_is_0_y_is_1_batch = rot_x_is_0_y_is_1_batch.to(device, non_blocking=True)
                        rot_ang_0_is_0_180_is_1_batch = rot_ang_0_is_0_180_is_1_batch.to(device, non_blocking=True)
                        
                        
                        image_batch_2 = image_batch_2.to(device, non_blocking=True)
                        reduced_image_batch_2 = reduced_image_batch_2.to(device, non_blocking=True) # Apply distortion
                        rot_x_is_0_y_is_1_batch_2 = rot_x_is_0_y_is_1_batch_2.to(device, non_blocking=True)
                        rot_ang_0_is_0_180_is_1_batch_2 = rot_ang_0_is_0_180_is_1_batch_2.to(device, non_blocking=True)
                        
                        with torch.cuda.amp.autocast():
                        
                            rot_prd,rot_axis_prd, contrastive_prd, recon_prd, rot_w, rot_axis_w, contrastive_w, recons_w = model(reduced_image_batch)
                            rot_prd_2,rot_axis_prd_2, contrastive_prd_2, recon_prd_2, _, _, _, _                         = model(reduced_image_batch_2)
                                
                            rot_p = torch.cat([rot_prd, rot_prd_2], dim=0) 
                            rots = torch.cat([rot_ang_0_is_0_180_is_1_batch, rot_ang_0_is_0_180_is_1_batch_2], dim=0) 
                            
                            rot_axis_p = torch.cat([rot_axis_prd, rot_axis_prd_2], dim=0) 
                            rots_axis = torch.cat([rot_x_is_0_y_is_1_batch, rot_x_is_0_y_is_1_batch_2], dim=0) 
                            
                            imgs_recon = torch.cat([recon_prd, recon_prd_2], dim=0) 
                            imgs = torch.cat([image_batch, image_batch_2], dim=0) 
                            
                            loss, (loss1, loss2, loss3, loss4) = criterion(rot_p, rots, 
                                                                        rot_axis_p, rots_axis,
                                                                        contrastive_prd, contrastive_prd_2, 
                                                                        imgs_recon, imgs, rot_w, rot_axis_w, contrastive_w, recons_w )
                                                                        
                        val_losses.update(loss.mean().item(), args.batch_size)
                            
                        writer.add_scalar('info/loss_rotation', loss1, iter_num)
                        writer.add_scalar('info/loss_rot_axis', loss2, iter_num)
                        writer.add_scalar('info/loss_contrastive', loss3, iter_num)
                        writer.add_scalar('info/loss_reconstruction', loss4, iter_num)
                    else:
                        (image_batch, proj_mask, label_batch, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) =  batch_data               
                        image_batch, label_batch = image_batch.cuda(), label_batch.cuda(non_blocking=True).long()
                        outputs = model(image_batch)
                        
                        loss_ce = ce_loss(outputs, label_batch)
                        
                        
                        val_losses.update(loss_ce.mean().item(), args.batch_size)
                        
                        argmax = outputs.argmax(dim=1)
                        
                        evaluator.addBatch(argmax, label_batch)
                        
                if not args.pretrain:    
                    jaccard, class_jaccard = evaluator.getIoU()
                    
                    iou.update(jaccard.item(), args.batch_size)#in_vol.size(0)) 

            if not args.pretrain: 
                writer.add_scalar('valid/iou', iou.avg, epoch_num)
                
            writer.add_scalar('valid/loss', val_losses.avg, epoch_num)
                
            
        ##############################################
        ##############################################

            
        save_interval = 30  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"