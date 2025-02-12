//===- Kernels.h - ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSMM_KERNELS_KERNELS_H
#define XSMM_KERNELS_KERNELS_H

#include "XsmmKernels.h"

enum class ComputeType { GEMM, BRGEMM };

enum class DataType { F32, BF16 };

bool isConfigSupported(ComputeType comp, DataType data, unsigned m, unsigned n,
                       unsigned k);

#endif // XSMM_KERNELS_KERNELS_H
