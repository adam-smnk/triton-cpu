#include "cpu/include/TritonToLinalg/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTTRITONTOLINALG
#include "cpu/include/TritonToLinalg/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct ConvertTritonToLinalg
    : public triton::cpu::impl::ConvertTritonToLinalgBase<
          ConvertTritonToLinalg> {
  using ConvertTritonToLinalgBase::ConvertTritonToLinalgBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTritonElementwiseToLinalgPatterns(patterns);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};

} // namespace
