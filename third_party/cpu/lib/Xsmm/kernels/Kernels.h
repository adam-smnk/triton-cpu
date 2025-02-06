//===- Kernels.h - ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSMM_KERNELS_KERNELS_H
#define XSMM_KERNELS_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "XsmmKernels.h"

#ifdef __cplusplus
} /* extern "c" */
#endif

bool isConfigSupported(unsigned m, unsigned n, unsigned k, unsigned batch = 0);

#endif // XSMM_KERNELS_KERNELS_H