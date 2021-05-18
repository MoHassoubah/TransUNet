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
import cv2
from matplotlib import pyplot as plt



def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)

def make_log_img(depth_gt, depth_gt_reduced, depth_pred, mask, mask_reduced, pred, color_fn,  gt, pretrain=False):
    
    if pretrain:
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth_gt = (cv2.normalize(depth_gt, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        out_img = cv2.applyColorMap(
            depth_gt, get_mpl_colormap('viridis')) * mask[..., None] 
            
        depth_gt_reduced = (cv2.normalize(depth_gt_reduced, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        depth_reduced_img = cv2.applyColorMap(
            depth_gt_reduced, get_mpl_colormap('viridis')) * mask_reduced[..., None]  
            
        out_img = np.concatenate([out_img, depth_reduced_img], axis=0)
        
            
        depth_pred = (cv2.normalize(np.float32(depth_pred), None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        depth_pred_img = cv2.applyColorMap(
            depth_pred, get_mpl_colormap('viridis')) * mask[..., None]  
            
        out_img = np.concatenate([out_img, depth_pred_img], axis=0)
    
    else:
        # make label prediction
        # pred_color = color_fn((pred * mask).astype(np.int32))
            
        out_img = color_fn((pred * mask).astype(np.int32))#np.concatenate([out_img, pred_color], axis=0)
        
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        
    return (out_img).astype(np.uint8)
    
def save_img(depth_gt, depth_gt_reduced, depth_pred, proj_mask, proj_mask_reduced, seg_outputs, proj_labels, parser_to_color, i_iter, pretrain=False):
    
    SAVE_PATH_kitti = '../result_train'
    if not pretrain:
        output = seg_outputs[0].permute(1,2,0).cpu().numpy()
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        
        
        mask_np = proj_mask[0].cpu().numpy()
        gt_np = proj_labels[0].cpu().numpy()
        out = make_log_img(None,None,None,mask_np,None, output, parser_to_color, gt_np, pretrain=args.pretrain )
    else:
        depth_gt_np = depth_gt[0][0].cpu().numpy()
        depth_gt_reduced_np = depth_gt_reduced[0][0].cpu().numpy()
        depth_pred_np = depth_pred[0][0].cpu().numpy()
        mask_np = proj_mask[0].cpu().numpy()
        mask_np_reduced =proj_mask_reduced[0].cpu().numpy()
        out = make_log_img(depth_gt_np, depth_gt_reduced_np, depth_pred_np, mask_np, mask_np_reduced, None, parser_to_color, None, pretrain=pretrain )
        
    # print(name)
    name_2_save = os.path.join(SAVE_PATH_kitti, '_'+str(i_iter) + '.png')
    cv2.imwrite(name_2_save, out)
        
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
    dir_path = snapshot_path+'/log'

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
                        
                    rot_p = torch.cat([rot_prd, rot_prd_2], dim=0).squeeze(1)
                    rots = torch.cat([rot_ang_0_is_0_180_is_1_batch, rot_ang_0_is_0_180_is_1_batch_2], dim=0) 
                    rots = rots.type_as(rot_p)
                    
                    # print("target rots")
                    # print(rots.shape)
                    # print(rots.dtype)
                    
                    # print("predicted rots")
                    # print(rot_p.shape)
                    
                    rot_axis_p = torch.cat([rot_axis_prd[rot_ang_0_is_0_180_is_1_batch==1], rot_axis_prd_2[rot_ang_0_is_0_180_is_1_batch_2==1]], dim=0).squeeze(1) 
                    rots_axis = torch.cat([rot_x_is_0_y_is_1_batch[rot_ang_0_is_0_180_is_1_batch==1], rot_x_is_0_y_is_1_batch_2[rot_ang_0_is_0_180_is_1_batch_2==1]], dim=0) 
                    rots_axis = rots_axis.type_as(rot_axis_p)
                    
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
                #loss1->rotation loss
                #loss2->rotation axis loss
                #loss3->contrastive loss
                #loss4->reconstruction loss
                logging.info('iteration %d : loss : %f, loss1 : %f, loss2 : %f, loss3 : %f, loss4 : %f' % (iter_num, loss.item(), \
                loss1.item(), loss2.item(), loss3.item(), loss4.item()))
                
                # logging.info('iteration %d : rot_axis_w : %f, contrastive_w : %f, recons_w : %f' % (iter_num, \
                # rot_axis_w.item(), contrastive_w.item(), recons_w.item()))
                
                # with torch.no_grad():
                    # if iter_num % 50 == 0 and iter_num != 0:
                        # save_img(image_batch, reduced_image_batch, recon_prd, proj_mask, reduced_proj_mask, None, None, parser.to_color, iter_num, pretrain=True)
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
                
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss', loss, iter_num)

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
        if args.pretrain:
            valid_interval=5
        else:
            valid_interval=10
            
        if (epoch_num + 1) % valid_interval == 0:
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
                                
                            rot_p = torch.cat([rot_prd, rot_prd_2], dim=0).squeeze(1)
                            rots = torch.cat([rot_ang_0_is_0_180_is_1_batch, rot_ang_0_is_0_180_is_1_batch_2], dim=0) 
                            rots = rots.type_as(rot_p)
                            
                            
                            rot_axis_p = torch.cat([rot_axis_prd[rot_ang_0_is_0_180_is_1_batch==1], rot_axis_prd_2[rot_ang_0_is_0_180_is_1_batch_2==1]], dim=0).squeeze(1) 
                            rots_axis = torch.cat([rot_x_is_0_y_is_1_batch[rot_ang_0_is_0_180_is_1_batch==1], rot_x_is_0_y_is_1_batch_2[rot_ang_0_is_0_180_is_1_batch_2==1]], dim=0) 
                            rots_axis = rots_axis.type_as(rot_axis_p)
                            
                            imgs_recon = torch.cat([recon_prd, recon_prd_2], dim=0) 
                            imgs = torch.cat([image_batch, image_batch_2], dim=0) 
                            
                            loss, (loss1, loss2, loss3, loss4) = criterion(rot_p, rots, 
                                                                        rot_axis_p, rots_axis,
                                                                        contrastive_prd, contrastive_prd_2, 
                                                                        imgs_recon, imgs, rot_w, rot_axis_w, contrastive_w, recons_w )
                                                                        
                        val_losses.update(loss.mean().item(), args.batch_size)
                            
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

            
        if args.pretrain:
            save_interval = 5  # int(max_epoch/6)
        else:
            save_interval = 10 
                
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