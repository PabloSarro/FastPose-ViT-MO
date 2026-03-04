#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Build the CUDA extension
python3 setup.py build_ext --inplace

# Move the generated .so file to the parent directory (LWDETR)
mv MultiScaleDeformableAttention.cpython-312-x86_64-linux-gnu.so ../