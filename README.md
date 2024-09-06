# MSDeformAttn : a CUDA Implementation of Multi-Scale Deformable Attention

## About

This is a standlone modernization of the Multi-Scale Deformable Attention implementation
from [facebookresearch's Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main).
This code might be useful to you if you are trying to replicate the experiments done
in the [DINOv2](https://github.com/facebookresearch/dinov2) repository on training
Mask2Former with a VIT backbone.

I modernized the implementation following this [guide](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#custom-ops-landing-page) from PyTorch.

## Features

- custom `torch.ops` for `forward` and `backward`
- a `torch.autograd.Function`
- a `nn.Module` conveniently packaging everything

## Installation

You just need to have CUDA installed on your machine.
Then simply do a `pip install .` .
