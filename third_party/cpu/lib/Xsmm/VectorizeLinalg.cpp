//===- VectorizeLinalg.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cpu/include/Xsmm/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_VECTORIZELINALG
#include "cpu/include/Xsmm/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

// Vectorize linalg ops targeting GPU.
struct LinalgVectorizer : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (linalgOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Expects static shapes only");

    return linalg::vectorize(rewriter, linalgOp);
  }
};

// Vectorize operations targeting GPU.
struct VectorizeLinalg
    : public triton::cpu::impl::VectorizeLinalgBase<VectorizeLinalg> {
  using VectorizeLinalgBase::VectorizeLinalgBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<LinalgVectorizer>(ctx);

    // Vector postprocessing patterns.
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);

    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
