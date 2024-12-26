# LoRAStencil

> LoRAStencil: Low-Rank Adaptation of Stencil Computation on Tensor Cores

## Abstract

This artifact contains the source code of LoRAStencil, a novel stencil computing system designed to mitigate memory access redundancies on TCUs through low-rank adaptation.

## Prerequisites

- Hardware
  - x86-64 CPU
  - NVIDIA A100 GPU
- Software
  - CUDA - 12.2 (Tested). Lower versions down to CUDA 11.0 are also supported, but it may affect the performance.
  - GCC - above 9.4.0. You may also try to use icx or clang.

## Getting Code

The code can be downloaded using git:
```
git clone https://github.com/HPHEX/LoRAStencil.git
```

## Compile

Use the following commands:
```
mkdir -p build
cd build
cmake ..
make all -j24
```

## Usage

You can run `lorastencil` in the following input format.

```
lorastencil_{x}d shape input_size time_size
```
- `lorastencil_{x}d` can be chosen from `lorastencil_1d`, `lorastencil_2d`, and `lorastencil_3d` for different dimensions.
- `shape` can be chosen by the different dimension:
  - `1d1r` and `1d2r` for 1D
  - `star2d1r`, `box2d1r`, `star2d3r` and `box2d3r` for 2D
  - `star3d1r` and `box3d1r` for 3D
- `input_size` depends on the number of dimensions, with the number of inputs equals the value of `x` in `lorastencil_{x}d`.
- `time_size` is the iteration time number.

## Contact
If you have any questions, please email to the corresponding author at [kunli@microsoft.com](kunli@microsoft.com).

## Reference

Yiwei Zhang, Kun Li, Liang Yuan, Jiawen Cheng, Yunquan Zhang, Ting Cao and Mao Yang. ["LoRAStencil: Low-Rank Adaptation of Stencil Computation on Tensor Cores."](https://dl.acm.org/doi/abs/10.1109/SC41406.2024.00059) In International Conference for High Performance Computing, Networking, Storage and Analysis (SC'24), pp. 839-855. IEEE Computer Society, 2024.

If you use our code, please cite our paper:

```angular2html
@inproceedings{10.1109/SC41406.2024.00059,
author = {Zhang, Yiwei and Li, Kun and Yuan, Liang and Cheng, Jiawen and Zhang, Yunquan and Cao, Ting and Yang, Mao},
title = {LoRAStencil: Low-Rank Adaptation of Stencil Computation on Tensor Cores},
year = {2024},
isbn = {9798350352917},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/SC41406.2024.00059},
doi = {10.1109/SC41406.2024.00059},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis},
articleno = {53},
numpages = {17},
location = {Atlanta, GA, USA},
series = {SC '24}
}
```
