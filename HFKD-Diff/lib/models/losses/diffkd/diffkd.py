import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline,CrossAttention
from .scheduling_ddim import DDIMScheduler
from .utils import GAP1d, get_module_dict, init_weights, is_cnn_model, PatchMerging, SepConv, set_module_dict, \
    TokenFilter, TokenFnContext, Patching,CNN_Trans, DISTILL_DIM_INFO
from timm.models.vision_transformer import Block
from vision.CAK import cka_heatmap


class OFA(nn.Module):
    requires_feat = True

    def __init__(self, student, feature_dim_s, feature_dim_t, student_is_cnn=True,student_name = None,teacher_name = None):
        super(OFA, self).__init__()

        is_cnn_student = student_is_cnn
        
        if is_cnn_student:
            self.projector = CNN_Trans(in_dim=feature_dim_s,out_dim= feature_dim_t, 
                                       model_name=student_name, model_t=teacher_name)

        else:
            patch_num, embed_dim = DISTILL_DIM_INFO[student_name]
            token_num = getattr(student, 'num_tokens', 0)  # cls tokens

            final_patch_grid = 7  # finally there are 49 patches
            patch_grid = int(patch_num ** .5)
            merge_num = max(int(np.log2(patch_grid / final_patch_grid)), 0)
            merger_modules = []
            for i in range(merge_num):
                if i == 0:  # proj to feature_dim_s
                    merger_modules.append(
                        PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                        dim=embed_dim,
                                        out_dim=feature_dim_t,
                                        model_name=student_name,
                                        model_t= teacher_name,
                                        act_layer=nn.GELU))
                else:
                    merger_modules.append(
                        PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                        dim=feature_dim_s,
                                        out_dim=feature_dim_t,
                                        model_name=student_name,
                                        model_t=teacher_name,
                                        act_layer=nn.GELU if i != merge_num - 1 else nn.Identity))
            if len(merger_modules) == 0:
                merger_modules.append(
                    Patching(dim=feature_dim_s,
                                        out_dim=feature_dim_t,
                                        model_name=student_name,
                                        act_layer=nn.GELU)
                )
            patch_merger = nn.Sequential(*merger_modules)
            self.projector = nn.Sequential(
                TokenFnContext(token_num, patch_merger),
            )
        self.projector.apply(init_weights)
        # print(self.projector)  # for debug

    def forward(self, feat):
        return self.projector(feat)



class channel_trans(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels):
        super().__init__()
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,
                             groups=groups)
        def conv5x5(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=stride, bias=False,
                             groups=groups)
        setattr(self, 'transfer1', nn.Sequential(
                conv1x1(student_channels, teacher_channels),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
                conv3x3(teacher_channels, teacher_channels),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
                conv1x1(teacher_channels, teacher_channels),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
            ))
        setattr(self, 'transfer2', nn.Sequential(
                conv1x1(student_channels, teacher_channels),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
                conv5x5(teacher_channels, teacher_channels),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
                conv1x1(teacher_channels, teacher_channels),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
            ))
    def forward(self, feat_s):
        trans_feat1 = getattr(self, "transfer1")(feat_s)
        trans_feat2 = getattr(self, "transfer2")(feat_s)
        return trans_feat1 + trans_feat2

def check_tuple(x):
    relu = nn.ReLU(inplace=False)
    contains_tensors=False
    if isinstance(x, tuple):
        contains_tensors = all(isinstance(item, torch.Tensor) for item in x)
    if contains_tensors:
            x = x[0]
            x = relu(x)
    return x 


class StudentProj(nn.Module):
    def __init__(self, teacher_name, student_name, latent_channel, cross_head):
        # Code collation in progress
    def forward(self, s):
        # Code collation in progress
        return
   
           
class DiffKD(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            student,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
            student_is_cnn = True,
            teacher_name = None,
            student_name = None,
            cross_head = 4
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels,teacher_name,student_name,cross_head)
            teacher_channels = ae_channels
            self.trans = StudentProj(teacher_name, student_name, teacher_channels,cross_head)
        
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        N, C = DISTILL_DIM_INFO[teacher_name]
        H = W = int(math.sqrt(N))
        if H * W != N:
            self.trans_t = nn.Sequential(nn.Conv1d(N, H*W, 1), nn.BatchNorm1d(H*W))
    def forward(self, student_feat, teacher_feat):
        # project student feature to the same dimension as teacher feature
        student_feat = check_tuple(student_feat)
        teacher_feat = check_tuple(teacher_feat)
        if student_feat.dim() == 2:
            student_feat = student_feat.view(student_feat.shape[0], student_feat.shape[1], 1, 1)
        else:
            student_feat = self.trans(student_feat)
        
        if teacher_feat.dim() == 2:
            teacher_feat = teacher_feat.view(teacher_feat.shape[0], teacher_feat.shape[1], 1, 1)
        elif teacher_feat.dim() == 3:
               # swin [B, N, C]
            B, N, C = teacher_feat.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                teacher_feat = self.trans_t(teacher_feat)
            teacher_feat = teacher_feat.permute(0, 2, 1).contiguous()
            teacher_feat = teacher_feat.view(B, C, H, W)
         
        # use autoencoder on teacher feature
        if self.use_ae:

            hidden_t_feat, rec_t_feat = self.ae(teacher_feat, student_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=None
        )
        
        # train diffusion model
        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
