import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def restore_and_save_images(output, epoch, batch_size, patch_size=(30, 30), num_patches=60, original_shape=(9, 30, 30), target=False):
    #output = output.reshape(batch_size, num_patches, *original_shape)
    output = output[:, :, :3, :, :]
    num_patches = output.shape[1]
    rows = int(num_patches**0.5)
    cols = num_patches // rows
    #print(rows,cols)
    
    for batch in range(batch_size):
        combined_image = torch.zeros((3, rows * patch_size[0], cols * patch_size[1]))
        
        for i in range(rows):
            for j in range(cols):
                patch_idx = i * cols + j
                patch = output[batch, patch_idx]
                combined_image[:, i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]] = patch
        
        unnorm = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])
        combined_image = unnorm(combined_image)
        combined_image = torch.clamp(combined_image, 0, 1)
        
        pil_image = transforms.ToPILImage()(combined_image)
        if target:
            filename = f"output/images/epoch_{epoch}_batch_{batch}_true.jpg"
        else:
            filename = f"output/images/epoch_{epoch}_batch_{batch}_pred.jpg"
        pil_image.save(filename)

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        #print("batch:", batch.keys())
        # videos, bool_masked_pos = batch
        #videos_input= batch["images"]
        # videos_input= batch["patches"]
        #videos= torch.unsqueeze(batch["images"], 1)
        # videos= torch.unsqueeze(batch["patches"], 1)
        # videos = videos.to(device, non_blocking=True)
        videos = batch["patches"].to(device)
        batch["ray_origins"],batch["ray_directions"] = batch["ray_origins"].to(device),batch["ray_directions"].to(device)
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        #bool_masked_pos = torch.rand(32).bool()
        # patch size = 30, patch num = 60
        #print(videos.shape)
        B,_,_,_,_ = videos.shape
        bool_masked_pos = (torch.rand(60)>0.5).repeat(B,1) 

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device).view(1, 1, 3, 1, 1)
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device).view(1, 1, 3, 1, 1)
            # print("mean:", mean)
            # print("std:", std)
            videos_patch = (videos - mean) / std
            # unnorm_videos = videos * std + mean  # in [0, 1]
#             unnorm_videos = videos

#             if normlize_target:
#                 videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
#                 videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
#                     ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
#                 # we find that the mean is about 0.48 and standard deviation is about 0.08.
#                 videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
#             else:
#                 videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            # B, _, C = videos_patch.shape
            # print("videos_patch.shape:", videos_patch.shape)
            # labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
            labels = torch.cat([videos_patch,batch["ray_origins"],batch["ray_directions"]],dim=2)[bool_masked_pos].flatten(start_dim=1)

        with torch.cuda.amp.autocast():
            rays = batch["ray_directions"]
            #print("rays.shape", rays.shape)
            ray_origins = batch["ray_origins"]
            outputs = model(videos_patch, ray_origins=ray_origins, ray_directions=rays, mask=bool_masked_pos)
            #print(labels.shape)
            #B, P, C = labels.shape
            loss = loss_func(input=outputs[bool_masked_pos].reshape(B,-1,9,30,30), target=labels.reshape(B,-1,9,30,30))
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
        #保存patch
        if epoch % 100 == 0:
            restore_and_save_images(outputs[bool_masked_pos].reshape(B,-1,9,30,30), epoch, batch_size=32,target=False)
            restore_and_save_images(labels.reshape(B,-1,9,30,30), epoch, batch_size=32,target=True)

                
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
