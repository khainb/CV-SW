Python3 implementation of the papers [Sliced Wasserstein Estimation with Control Variates](https://arxiv.org/abs/2305.00402)

Details of the model architecture and experimental results can be found in our papers.


## Requirements
The code is implemented with Python (3.8.8) and Pytorch (1.9.0).

## What is included?
* Control Variate Sliced Wasserstein Generators

## Sliced Wasserstein Generators
### Code organization
* cfg.py : this file contains arguments for training.
* datasets.py : this file implements dataloaders.
* functions.py : this file implements training functions.
* trainsw.py : this file is the main file for running.
* models : this folder contains neural networks architecture.
* utils : this folder contains implementation of fid score and Inception score.
* fid_stat : this folder contains statistic files for fID score.

### Script examples
CIFAR10
```
GPU=1
seed=1
L=10
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset cifar10 --img_size 32 --max_iter 100000 --model sngan_cifar10 --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 50 --L ${L} --sw_type sw --random_seed ${seed} --exp_name SW_L${L}_seed${seed}
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset cifar10 --img_size 32 --max_iter 100000 --model sngan_cifar10 --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 50 --L ${L} --sw_type lcvsw --random_seed ${seed} --exp_name GCSW_L${L}_seed${seed}
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset cifar10 --img_size 32 --max_iter 100000 --model sngan_cifar10 --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 50 --L ${L} --sw_type ucvsw --random_seed ${seed} --exp_name UGCSW_L${L}_seed${seed}
```

CelebA
```
GPU=1
seed=1
L=10
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset celeba --img_size 64 --max_iter 50000 --model sngan_celeba --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --L ${L} --sw_type sw --random_seed ${seed} --exp_name SW_L${L}_seed${seed}
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset celeba --img_size 64 --max_iter 50000 --model sngan_celeba --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --L ${L} --sw_type lcvsw --random_seed ${seed} --exp_name GCSW_L${L}_seed${seed}
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset celeba --img_size 64 --max_iter 50000 --model sngan_celeba --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --L ${L} --sw_type ucvsw --random_seed ${seed} --exp_name UGCSW_L${L}_seed${seed}
```

## Acknowledgment
The structure of this repo is largely based on [sngan.pytorch](https://github.com/GongXinyuu/sngan.pytorch).