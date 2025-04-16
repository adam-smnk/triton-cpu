#include "cpu/include/TritonToLinalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Based on triton-shared converter.
struct ReduceConverter : public OpRewritePattern<triton::ReduceOp> {
  using OpRewritePattern<triton::ReduceOp>::OpRewritePattern;

private:
  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const {
    auto reduceBlock = redOp.getBody();
    return llvm::map_to_vector(reduceBlock->without_terminator(),
                                [](Operation &op) { return &op; });
  }

  bool isReductionOpSupported(Operation *redOp) const {
    return isa<arith::AddFOp, arith::AddIOp, arith::MaximumFOp,
                arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
                arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>(
        redOp);
  }

  arith::ConstantOp getRedBaseConstOp(PatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr =
        llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
            .Case([&](arith::AddFOp) {
              return rewriter.getFloatAttr(constantType, 0.f);
            })
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, -std::numeric_limits<float>::infinity());
            })
            .Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, std::numeric_limits<float>::infinity());
            })
            .Case([&](arith::MinSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                              llvm::maxIntN(bitWidth));
            })
            .Case([&](arith::MinUIOp) {
              return rewriter.getIntegerAttr(constantType,
                                              llvm::maxUIntN(bitWidth));
            })
            .Case([&](arith::MaxSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                              llvm::minIntN(bitWidth));
            })
            .Case([&](arith::MaxUIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Default([](Operation *op) {
              op->dump();
              llvm_unreachable("Reduction op not yet supported");
              return nullptr;
            });

    return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType,
                                              attr);
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision = false) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case([&](arith::AddFOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<arith::AddFOp>(loc, lhs, rhs);
        })
        .Case<arith::AddIOp, arith::MaximumFOp, arith::MaxNumFOp,
              arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
              arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>([&](auto redOp) {
          return b.create<decltype(redOp)>(loc, lhs, rhs);
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
    PatternRewriter &rewriter) const {
    auto source = op.getOperands().front();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto elemType = sourceType.getElementType();
    auto resType = op.getResult().front().getType();
    auto loc = op.getLoc();
    auto reductionOps = getRedOps(op);

    // Reduction of arbitrary operations isn't supported because using the first
    // element across the reduction dimension requires us to iterate over a
    // subview that skips over each first element.
    if (reductionOps.size() != 1 ||
        !isReductionOpSupported(reductionOps.front())) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering reduction with body "
              "containing 1 max(i/f) or addf.");
    }

    auto rop = reductionOps.front();
    auto axis = op.getAxis();

    auto constantType = elemType;

    auto accBaseConstOp = getRedBaseConstOp(rewriter, rop, constantType);
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(resType).getShape(), constantType);
    Value initTensor = rewriter
                      .create<linalg::FillOp>(loc, ValueRange{accBaseConstOp},
                                              ValueRange{init})
                      .result();

    Value finalResult =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{source}, ValueRange{initTensor},
                SmallVector<int64_t>{axis},
                [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                  assert(inputs.size() == 2);
                  Value result =
                      getRedElement(inputs[1], inputs[0], loc, rop, opBuilder);
                  opBuilder.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);

    if (sourceType.getRank() == 1) {
      finalResult =
          rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp reduceOp, PatternRewriter &rewriter) const override {
    if(reduceOp.getNumOperands() != 1)
      return rewriter.notifyMatchFailure(
        reduceOp, "Requires single input");

    auto sourceType = dyn_cast<RankedTensorType>(reduceOp.getSrcs().front().getType());
    if(!sourceType)
      return rewriter.notifyMatchFailure(
        reduceOp, "Requires ranked tensor input");

    int64_t axis = reduceOp.getAxis();
    if(!(axis >= 0 && axis < sourceType.getRank()))
      return rewriter.notifyMatchFailure(
        reduceOp, "Invalid reduction axis");

    return convertToLinalgReduce(reduceOp, rewriter);
  }
};

} // namespace

void mlir::triton::cpu::populateTritonReduceToLinalgPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ReduceConverter>(patterns.getContext());
}
