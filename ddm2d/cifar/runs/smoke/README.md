# CPU smoke (2026-11-26, X2): proves the code path, not a result

`bash cifar/scripts/smoke_cpu.sh` (from `ddm2d/`): every run config with a tiny U-Net (16 base channels, 2 levels), 20 steps on the
first 512 train images (batch 8, C_dino 4; grad_chunk 12 to exercise the exact two-pass gradient), then NFE-2 sampling of 300 images,
a 300-image clean-fid call (legacy_tensorflow, CIFAR-10 train stats), a 10 x 10 grid and the xi-spread. C_dino used the real DINOv2
ViT-S/14 on CPU. FID values here (~400) are meaningless and must never appear in a result table.
