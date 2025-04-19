<!-- # DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers <br><sub>Official PyTorch Implementation</sub> -->

# DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers

<!-- [![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.11838-b31b1b.svg)](https://arxiv.org/abs/2406.11838)&nbsp; -->
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-image-generation-without/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=autoregressive-image-generation-without) -->
<!-- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/LTH14/mar/blob/main/demo/run_mar.ipynb) -->
<!-- [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-mar-yellow)](https://huggingface.co/jadechoghari/mar)&nbsp; -->


<div align="center">
<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<img src="figs/logo.png" width="65%"/>
</div>

### [<a href="https://huggingface.co/papers/2503.14487" target="_blank">arXiv</a>] [<a href="https://shiml20.github.io/DiffMoE/" target="_blank">Project Page</a>] [<a href="https://huggingface.co/datasets/KwaiVGI/DiffMoE/" target="_blank">Code</a>]

_**[Minglei Shi<sup>1*</sup>](https://github.com/shiml20/), [Ziyang Yuan<sup>1*</sup>](https://scholar.google.ru/citations?user=fWxWEzsAAAAJ&hl=en), [Haotian Yang<sup>2</sup>](https://scholar.google.com/citations?user=LH71RGkAAAAJ&hl=en), [Xintao Wang<sup>2†</sup>](https://xinntao.github.io/),  [Mingwu Zheng<sup>2</sup>](https://scholar.google.com.hk/citations?user=MdizB60AAAAJ&hl=en), [Xin Tao<sup>2</sup>](https://www.xtao.website/), [Wenliang Zhao<sup>1</sup>](https://wl-zhao.github.io/), [Wenzhao Zheng<sup>1</sup>](https://wzzheng.net/), [Jie Zhou<sup>1</sup>](https://www.imem.tsinghua.edu.cn/info/1330/2128.htm),[Jiwen Lu<sup>1†</sup>](https://www.au.tsinghua.edu.cn/info/1078/3156.htm), [Pengfei Wan<sup>2</sup>](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en), [Di Zhang<sup>2</sup>](https://openreview.net/profile?id=~Di_ZHANG3), [Kun Gai<sup>2</sup>](https://scholar.google.com/citations?user=PXO4ygEAAAAJ&hl=zh-CN)**_
<br>
(*equal contribution, †corresponding author.)

<sup>1</sup>Tsinghua University, <sup>2</sup>Kuaishou Technology.

</div>


## 🔥 Updates
- __[2025.3.20]__: Release the [code](https://github.com/KwaiVGI/DiffMoE/) of DiffMoE
- __[2025.3.19]__: Release the [project page](https://shiml20.github.io/DiffMoE/) of DiffMoE

<!-- This is a PyTorch/GPU implementation of the paper [DiffMoE](xxxx) -->


## 📖 Introduction

**TL;DR:** DiffMoE is a dynamic MoE Transformer that outperforms 3× larger dense models in diffusion tasks, using global token pool and adaptive routing while keeping 1× parameter activation.  <br>


This repo contains:

* 🪐 A simple PyTorch implementation of [Dense-DiT](models/models_DiT.pu), [EC-DiT](models/models_ECDiT.py), [TC-DiT](models/models_TCDiT.py), [DifffMoE](models/models_DiffMoE.py)
<!-- * ⚡️ Pre-trained class-conditional DiffMoE models trained on ImageNet 256x256 and 512x512 -->
* 🛸 An DiffMoE [training and evaluation script](train.py) using PyTorch DDP
<!-- * 🎉 Also checkout our [Hugging Face model cards](xxxxx). -->


### To-do list

- [x] training / inference scripts
- [ ] huggingface ckpts

## ✨ Key Points


<p align="center">
  <img src="figs/teaser.png" width="720">
</p>

Token Accessibility and Dynamic Computation. (a) Token accessibility levels from token isolation to crosssample interaction. Colors represent tokens in different samples, ti indicates noise levels. (b) Performance-accessibility analysis across architectures. (c) Computational dynamics during diffusion sampling, showing adaptive computation from noise to image. (d) Class-wise computation allocation from hard (technical diagrams) to easy (natural photos) tasks. Results from DiffMoE-L-E16-Flow (700K).


<p align="center">
  <img src="figs/method.png" width="720">
</p>

DiffMoE Architecture Overview. DiffMoE flattens tokens into a batch-level global token pool, where each expert maintains a fixed training capacity of $C^{E_i}_{train} = 1$. During inference, a dynamic capacity predictor adaptively routes tokens across different sampling steps and conditions. Different colors denote tokens from distinct samples, while ti represents corresponding noise levels.



## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/KwaiVGI/DiffMoE.git 
cd DiffMoE
```


A suitable [conda](https://conda.io/) environment named `diffmoe` can be created and activated with:

```
conda env create -f environment.yaml
conda activate diffmoe
```

<!-- Download pre-trained VAE and DiffMoE models: -->

<!-- ``` -->
<!-- python util/download.py -->
<!-- ``` -->

Pre-trained models are Coming soon, stay tuned !


| Model Name                 | # Avg. Act. Params | Training Step | CFG | FID-50K↓  | Inception Score↑ |
|----------------------------|-------------------------|-------------|-----|---------|----------------|
| TC-DiT-L-E16-Flow   | 458M               | 700K        | 1.0 | 19.06   | 73.49           |
| EC-DiT-L-E16-Flow    | 458M               | 700K        | 1.0 | 16.12   | 82.37           |
| Dense-DiT-L-Flow     | 458M               | 700K        | 1.0 | 17.01   | 78.17           |
| Dense-DiT-XL-Flow   | 675M               | 700K        | 1.0 | 14.77   | 86.82           |
| DiffMoE-L-E16-Flow   | 454M               | 700K        | 1.0 | 14.41   | 88.19           |
| Dense-DiT-XL-Flow   | 458M                | 7000K       | 1.0 | 9.47 | 115.58         |
| DiffMoE-L-E8-Flow   | 458M                | 7000K       | 1.0 | 9.60 | 131.45         |
| Dense-DiT-XL-DDPM   | 458M                | 7000K       | 1.0 | 9.62 | 123.19         |
| DiffMoE-L-E8-DDPM   | 458M                | 7000K       | 1.0 | 9.17 | 131.10         |


<!-- ### (Optional) Caching VAE Latents

Given that our data augmentation consists of simple center cropping and random flipping, 
the VAE latents can be pre-computed and saved to `CACHED_PATH` to save computations during DiffMoE training:

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path stabilityai/sd-vae-ft-mse \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
``` -->

## Usage

<!-- ### Demo -->
<!-- Run our interactive visualization [demo](http://colab.research.google.com/github/LTH14/mar/blob/main/demo/run_mar.ipynb) using Colab notebook! -->
<!-- Comeing soon, stay tuned ! -->

<!-- ### Local Gradio App -->

<!-- Comeing soon, stay tuned ! -->

<!-- ```
python demo/gradio_app.py 
``` -->


### Training
Script for the default setting:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
train.py --config ./config/000_DiffMoE_S_E16_Flow.yaml
```

<!-- 
- Training time is ~1d7h on 32 H100 GPUs with `--batch_size 64`.
- Add `--online_eval` to evaluate FID during training (every 40 epochs).
- (Optional) To train with cached VAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments. 
Training time with cached latents is ~1d11h on 16 H100 GPUs with `--batch_size 128` (nearly 2x faster than without caching).
- (Optional) To save GPU memory during training by using gradient checkpointing (thanks to @Jiawei-Yang), add `--grad_checkpointing` to the arguments. 
Note that this may slightly reduce training speed. -->

### Evaluation (ImageNet 256x256)

Evaluate DiffMoE:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nnodes=1 --nproc_per_node=8 \
sample_ddp_feature.py --image-size 256 \
    --per-proc-batch-size 125 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir samples \
    --ckpt exps/EXPNAME/checkpoints/xxxxxx.pt
```



<!-- 
- Set `--cfg 1.0 --temperature 0.95` to evaluate without classifier-free guidance.
- Generation speed can be significantly increased by reducing the number of autoregressive iterations (e.g., `--num_iter 64`). -->

## Acknowledgements
We thank [Zihan, Qiu](https://scholar.google.com/citations?user=24eVHiYAAAAJ&hl=en) for helpful discussion. A large portion of codes in this repo is based on [MAR](https://github.com/LTH14/mar), [DiT](https://github.com/facebookresearch/DiT), [DeepSeekMoE](https://github.com/deepseek-ai/DeepSeek-MoE)

<!-- ## Contact -->


## 🌟 Citation

```
@misc{shi2025diffmoedynamictokenselection,
      title={DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers}, 
      author={Minglei Shi and Ziyang Yuan and Haotian Yang and Xintao Wang and Mingwu Zheng and Xin Tao and Wenliang Zhao and Wenzhao Zheng and Jie Zhou and Jiwen Lu and Pengfei Wan and Di Zhang and Kun Gai},
      year={2025},
      eprint={2503.14487},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.14487}, 
}
```