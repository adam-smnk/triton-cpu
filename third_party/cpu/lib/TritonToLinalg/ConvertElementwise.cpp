#include "cpu/include/TritonToLinalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  // TODO: The conversion pattern can be made to work for `any_of` here, but
  // it's more complex as it requires tracking which operands are scalars.
  return llvm::all_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>);
}

std::optional<linalg::ElementwiseKind> getElementwiseKind(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<linalg::ElementwiseKind>>(op)
      .Case<math::ExpOp>([&](auto op) { return linalg::ElementwiseKind::exp; })
      .Case<math::LogOp>([&](auto op) { return linalg::ElementwiseKind::log; })
      .Case<math::AbsFOp, math::AbsIOp>(
          [&](auto op) { return linalg::ElementwiseKind::abs; })
      .Case<math::CeilOp>(
          [&](auto op) { return linalg::ElementwiseKind::ceil; })
      .Case<math::FloorOp>(
          [&](auto op) { return linalg::ElementwiseKind::floor; })
      .Case<arith::NegFOp>(
          [&](auto op) { return linalg::ElementwiseKind::negf; })
      .Case<math::SqrtOp>(
          [&](auto op) { return linalg::ElementwiseKind::sqrt; })
      .Case<math::RsqrtOp>(
          [&](auto op) { return linalg::ElementwiseKind::rsqrt; })
      .Case<math::ErfOp>([&](auto op) { return linalg::ElementwiseKind::erf; })
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return linalg::ElementwiseKind::add; })
      .Case<arith::SubIOp, arith::SubFOp>(
          [&](auto op) { return linalg::ElementwiseKind::sub; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return linalg::ElementwiseKind::mul; })
      .Case<arith::DivSIOp, arith::DivFOp>(
          [&](auto op) { return linalg::ElementwiseKind::div; })
      .Case<arith::DivUIOp>(
          [&](auto op) { return linalg::ElementwiseKind::div_unsigned; })
      .Case<arith::DivUIOp>(
          [&](auto op) { return linalg::ElementwiseKind::div_unsigned; })
      .Case<arith::MaxSIOp, arith::MaximumFOp, arith::MaxNumFOp>(
          [&](auto op) { return linalg::ElementwiseKind::max_signed; })
      .Case<arith::MinSIOp, arith::MinimumFOp, arith::MinNumFOp>(
          [&](auto op) { return linalg::ElementwiseKind::min_signed; })
      .Case<arith::MaxUIOp>(
          [&](auto op) { return linalg::ElementwiseKind::max_unsigned; })
      .Case<arith::MinUIOp>(
          [&](auto op) { return linalg::ElementwiseKind::min_unsigned; })
      .Default([&](auto op) { return std::nullopt; });
}

struct ConvertElementwiseOnRankedTensors : public RewritePattern {
  ConvertElementwiseOnRankedTensors(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    if (!isElementwiseMappableOpOnRankedTensors(op))
      return rewriter.notifyMatchFailure(
          op, "requires elementwise op on ranked tensors");

    std::optional<linalg::ElementwiseKind> eltwiseKind = getElementwiseKind(op);
    if (!eltwiseKind)
      return rewriter.notifyMatchFailure(op,
                                         "could not map to elementwise kind");

    auto rank = cast<RankedTensorType>(op->getResult(0).getType()).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));

    ValueRange operands = op->getOperands();
    TypeRange resultTypes = op->getResultTypes();
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, operands.front()),
        cast<RankedTensorType>(resultTypes.front()).getElementType());

    rewriter.replaceOpWithNewOp<linalg::ElementwiseOp>(
        op, op->getOperands(), ValueRange{output},
        linalg::ElementwiseKindAttr::get(rewriter.getContext(), *eltwiseKind),
        rewriter.getAffineMapArrayAttr(indexingMaps));
    return success();
  }
};

} // namespace

void mlir::triton::cpu::populateTritonElementwiseToLinalgPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConvertElementwiseOnRankedTensors>(patterns.getContext());
}
