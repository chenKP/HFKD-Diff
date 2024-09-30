import math
import torch
from torch.nn import Linear
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from .kl_div import KLDivergence
from .dist_kd import DIST
from .diffkd import DiffKD
from .dkd import DKD
from timm.models.layers import _assert

from .diffkd.utils import DISTILL_DIM_INFO
from einops import rearrange
from torch import einsum
import logging
from lib.models.vit import Attention, CONFIGS
from vision.CAK import cka_heatmap
logger = logging.getLogger()



KD_MODULES = {
    'cifar_wrn_40_1': dict(modules=['relu', 'fc'], channels=[64, 100]),
    'cifar_wrn_40_2': dict(modules=['relu', 'fc'], channels=[128, 100]),

    'cifar_resnet56': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'cifar_resnet20': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'cifar_resnet32x4': dict(modules=['layer3', 'fc'], channels=[256, 100]),
    'cifar_resnet8x4': dict(modules=['layer3', 'fc'], channels=[256, 100]),
    'cifar_vgg13': dict(modules=['block4', 'classifier'], channels=[512, 100]),
    'cifar_vgg8': dict(modules=['block4', 'classifier'], channels=[512, 100]),
    'cifar_ShuffleV1': dict(modules=['layer3', 'linear'], channels=[960, 100]),
    'cifar_ShuffleV2': dict(modules=['layer3', 'linear'], channels=[464, 100]),
    'cifar_resnet50': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'cifar_mobile_half': dict(modules=['conv2', 'classifier'], channels=[1280, 100]),
    'resnet56': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'resnet20': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'mobilenet_v2': dict(modules=['features.18', 'classifier'], channels=[1280, 100]),

    "timm_resnet18":dict(modules=['layer4', 'fc'], channels=[512, 100]),
    "timm_swin_tiny_patch4_window7_224":dict(modules=['layers-3', 'head'], channels=[768, 100]),
    "timm_vit_small_patch16_224":dict(modules=['blocks-11', 'head'], channels=[384, 100]),
    "timm_mixer_b16_224":dict(modules=['blocks-11', 'head'], channels=[768, 100]),
    "timm_convnext_tiny":dict(modules=['stages-3', 'head-fc'], channels=[768, 100]),
    "timm_deit_tiny_patch16_224":dict(modules=['blocks-11', 'head'], channels=[192, 100]),
    "timm_swin_pico_patch4_window7_224":dict(modules=['layers-3', 'head'], channels=[384, 100]),
    "timm_resmlp_12_224":dict(modules=['blocks-11', 'head'], channels=[384, 100]),
    #=====================================================================
    'tv_resnet50': dict(modules=['layer4', 'fc'], channels=[2048, 1000]),
    'tv_resnet34': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_mobilenet_v2': dict(modules=['features.17', 'classifier'], channels=[1280, 1000]),
    'nas_model': dict(modules=['features.conv_out', 'classifier'], channels=[1280, 1000]),  
    'timm_tf_efficientnet_b0': dict(modules=['conv_head', 'classifier'], channels=[1280, 1000]),
    'mobilenet_v1': dict(modules=['model.13', 'fc'], channels=[1024, 1000]),
    'timm_swin_large_patch4_window7_224': dict(modules=['norm', 'head'], channels=[1536, 1000]),
    #'timm_swin_tiny_patch4_window7_224': dict(modules=['norm', 'head'], channels=[768, 1000]),
}


class MaskedFM(nn.Module):

    def __init__(self, heads=16, in_dim = 1024,student_name = None, teacher_name = None):
        super().__init__()
        # self.feature_pairs = feature_pairs
        self.student_name = student_name
        self.teacher_name = teacher_name
        self.heads=heads
        L,_ = DISTILL_DIM_INFO[student_name]
        L_T,_ = DISTILL_DIM_INFO[teacher_name]
        group = L_T // L
        if group == 0:
            group =  L // L_T
        self.reduction = nn.Sequential(nn.Linear(group* in_dim, in_dim, bias=False),
                                       nn.LayerNorm(in_dim, in_dim))
        self.trans_c = nn.Conv2d(group*in_dim, in_dim, 1,padding=0)
    def forward(self, s, t):
        loss1, s, t = self.channel_wise_means(s,t)
        return loss1, s, t
    def patch_feat(self, g, h, feat):
        # Code collation in progress
    
    def channel_wise_means(self, s, t):  
        # Code collation in progress
        
def batch_loss(heads, q, v, k, t):
    # Code collation in progress
    return

class MSELoss_modify(nn.Module):
    def __init__(self, heads = 1,  in_channels = 1024,student_name = None, teacher_name = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.heads = heads
        self.key = Linear(in_channels,heads * in_channels)
        self.layer_norm = torch.nn.LayerNorm(in_channels, eps=1e-6)
        self.mskloss = MaskedFM(heads=heads,in_dim=in_channels, student_name= student_name, teacher_name=teacher_name)
    def __call__(self, f_student, f_teacher, target_mask):
        # Code collation in progress
    
class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(
        self,
        student,
        teacher,
        student_name,
        teacher_name,
        ori_loss,
        kd_method='kdt4',
        ori_loss_weight=1.0,
        kd_loss_weight=1.0,
        kd_loss_kwargs={},
        assist_kd="dist",
        cross_head = 4,
        student_is_cnn = True,
        args = None,
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight

        self._teacher_out = None
        self._student_out = None
        self.assist_kd = assist_kd
        # init kd loss
        # module keys for distillation. '': output logits
        teacher_modules = ['',]
        student_modules = ['',]
        if kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
        elif kd_method == 'dkd':
            self.kd_loss = DKD(alpha=args.dkd_alpha, beta=args.dkd_beta, tau=4)
        elif kd_method.startswith('dist_t'):
            tau = float(kd_method[6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        
        elif kd_method == 'diffkd':
            # get configs
            ae_channels = kd_loss_kwargs.get('ae_channels', 1024)
            use_ae = kd_loss_kwargs.get('use_ae', True)
            tau = kd_loss_kwargs.get('tau', 1)

            print(kd_loss_kwargs)
            kernel_sizes = [3, 1]  # distillation on feature and logits
            student_modules = KD_MODULES[student_name]['modules']
            student_channels = KD_MODULES[student_name]['channels']
            teacher_modules = KD_MODULES[teacher_name]['modules']
            teacher_channels = KD_MODULES[teacher_name]['channels']
            self.diff = nn.ModuleDict()
            self.kd_loss = nn.ModuleDict()
            for tm, tc, sc, ks in zip(teacher_modules, teacher_channels, student_channels, kernel_sizes):
             
                self.diff[tm] = DiffKD(sc, tc, student, kernel_size=ks, use_ae=(ks!=1) and use_ae, ae_channels=ae_channels, 
                                       student_is_cnn=student_is_cnn,teacher_name=teacher_name, student_name=student_name,cross_head=cross_head
                                       ) # type: ignore

                if assist_kd == "dist":
                    self.kd_loss[tm] = MSELoss_modify(in_channels=ae_channels, student_name=student_name, teacher_name=teacher_name) \
                                    if ks != 1 else DIST(beta=args.dist_beta, gamma=args.dist_gamma, tau=4)
                elif assist_kd == "dkd":
                    self.kd_loss[tm] = MSELoss_modify(in_channels=ae_channels, student_name=student_name, teacher_name=teacher_name) \
                                    if ks != 1 else DKD(alpha=args.dkd_alpha, beta=args.dkd_beta, tau=4)        
                elif assist_kd == "mlkd":
                    from .MLKD import MLKD
                    self.kd_loss[tm] = MSELoss_modify(in_channels=ae_channels, student_name=student_name, teacher_name=teacher_name) \
                                    if ks != 1 else MLKD(ce_weight=args.mlkd_ce, kd_weight=args.mlkd_kd, tau=4)
                else:
                    self.kd_loss[tm] = MSELoss_modify(in_channels=ae_channels, student_name=student_name, teacher_name=teacher_name) \
                                    if ks != 1 else KLDivergence(tau=tau)
            self.diff.cuda()
            self.kd_loss.cuda()
            # add diff module to student for optimization
            self.student._diff = self.diff

        elif kd_method == 'mse':
            # distillation on feature
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            self.kd_loss = nn.MSELoss()
            self.align = nn.Conv2d(student_channels[0], teacher_channels[0], 1) # type: ignore
            self.align.cuda()
            # add align module to student for optimization
            self.student._align = self.align
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        # register forward hook
        # dicts that store distillation outputs of student and teacher
        self._teacher_out = {}
        self._student_out = {}

        for student_module, teacher_module in zip(student_modules, teacher_modules):
            self._register_forward_hook(student, student_module, teacher=False)
            self._register_forward_hook(teacher, teacher_module, teacher=True)
        self.student_modules = student_modules
        self.teacher_modules = teacher_modules

        teacher.eval()
        self._iter = 0

    def __call__(self, x, targets):
        with torch.no_grad():
            t_logits = self.teacher(x)

        # compute ori loss of student
        logits = self.student(x)
        ori_loss = self.ori_loss(logits, targets)

        kd_loss = 0

        for tm, sm in zip(self.teacher_modules, self.student_modules):
            s_out = sm
            if '-' in sm:
                    s_out = sm.replace("-", ".")
            input_s = self._student_out[s_out]
            # transform student feature
            t_out = tm
            if self.kd_method == 'diffkd':
                if '-' in tm:
                    t_out = tm.replace("-", ".")
                self._student_out[sm], self._teacher_out[tm], diff_loss, ae_loss = \
                    self.diff[tm](self._student_out[s_out], self._teacher_out[t_out])
            if hasattr(self, 'align'):
                self._student_out[sm] = self.align(self._student_out[sm])

            # compute kd loss
            if isinstance(self.kd_loss, nn.ModuleDict):

                target_mask = None
                num_classes = logits.size(-1)
                temp = self._student_out[sm].shape[1]
                if self._student_out[sm].shape[1] == num_classes and \
                    ((sm == "linear") or (sm == "fc") or (sm == "classifier") or (sm == "head") or (sm == "head-fc")):
                    if len(targets.shape) != 1:  # label smoothing
                        target_mask = F.one_hot(targets.argmax(-1), num_classes)
                    else:
                        target_mask = F.one_hot(targets, num_classes)
                if self.assist_kd == "dkd":
                    kd_loss_ = self.kd_loss[tm](self._student_out[sm], self._teacher_out[tm],targets)
                elif self.assist_kd == "mlkd":
                    kd_loss_ = self.kd_loss[tm](self._student_out[sm], self._teacher_out[tm],targets)
                else:
                    kd_loss_ = self.kd_loss[tm](self._student_out[sm], self._teacher_out[tm],None)
            else:
                # kd_loss_ = self.kd_loss(self._student_out[sm], self._teacher_out[tm],None)
                if self.kd_method == 'dkd':
                    kd_loss_ = self.kd_loss(logits, t_logits, targets)
                else:    
                    kd_loss_ = self.kd_loss(logits, t_logits, None)

            if self.kd_method == 'diffkd':
                # add additional losses in DiffKD
                if ae_loss is not None:
                    kd_loss += diff_loss + ae_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f} AE loss: {ae_loss.item():.4f}')
                else:
                    kd_loss += diff_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f}')
            else:
                if self._iter % 50 == 0:
                    logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f}')
            #=========================================================================================
            if self._student_out[sm].shape[1] == num_classes and \
                    ((sm == "linear") or (sm == "fc") or (sm == "classifier") or (sm == "head") or (sm == "head-fc")):
                kd_loss += kd_loss_
            else:
                kd_loss += kd_loss_  * self.kd_loss_weight

        self._teacher_out = {}
        self._student_out = {}

        self._iter += 1
        # if self.assist_kd == "mlkd":
        #     return ori_loss * 0.1 + kd_loss * self.kd_loss_weight
        return ori_loss * self.ori_loss_weight + kd_loss

    def _register_forward_hook(self, model, name, teacher=False):
        if '-' in name:
            name = name.replace("-", ".")
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            module.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))

    def _forward_hook(self, module, input, output, name, teacher=False):
        if '-' in name:
            name = name.replace("-", ".")
        if teacher:
            self._teacher_out[name] = output[0] if len(output) == 1 else output
        else:
            self._student_out[name] = output[0] if len(output) == 1 else output

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        contains_tensors=False
        if isinstance(x, tuple):
            contains_tensors = all(isinstance(item, torch.Tensor) for item in x)
        
        if contains_tensors:
             x = x[0]
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
               # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                trans = nn.Conv1d(N, H*W, 1)
                trans = trans.cuda()
                x = trans(x)
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x