from .diffkd import DiffKD

DISTILL_DIM_INFO = {
    "timm_mixer_b16_224":(196,768),
    "resmlp_12_224":(196, 384),
    "timm_swin_tiny_patch4_window7_224":(49,798),
    "swin_pico_patch4_window7_224":(49,384),
    "timm_vit_small_patch16_224":(197,384),
    "timm_deit_tiny_patch16_224":(197,192),
}