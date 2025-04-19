#include "cpu/include/TritonToLinalg/Passes.h"
#include "cpu/include/TritonToLinalg/TypeConverter.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTTRITONTOFUNC
#include "cpu/include/TritonToLinalg/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct ConvertTritonToFunc
    : public triton::cpu::impl::ConvertTritonToFuncBase<ConvertTritonToFunc> {
  using ConvertTritonToFuncBase::ConvertTritonToFuncBase;

  // Function conversion from triton-shared.
  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;
  // Add additional I32 arguments to represent:
  // - num_programs, 3 in total, one for each axis of the launch grid
  // - program_id, 3 in total, one for each axis of the launch grid
  static void addProgramInfo(triton::FuncOp func) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add empty attributes for each new argument if needed
    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    // Add the corresponding arguments to function body
    for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    TritonTypeConverter converter(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<memref::MemRefDialect, tensor::TensorDialect,
                           func::FuncDialect>();
    // Update function signature and call ops to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp, triton::FuncOp>([&](auto op) {
      return converter.isSignatureLegal(
          cast<FunctionType>(cast<FunctionOpInterface>(op).getFunctionType()));
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isLegal(op.getResultTypes()) &&
             converter.isLegal(op.getOperandTypes());
    });
    // config illegal ops

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(patterns,
                                                                     converter);
    populateCallOpTypeConversionPattern(patterns, converter);

    moduleOp.walk([&](triton::FuncOp func) { addProgramInfo(func); });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      return signalPassFailure();

    // Convert tt.func and tt.return into func's counterparts
    moduleOp.walk([&](triton::FuncOp func) {
      OpBuilder builder(func);

      auto name = func.getName();
      auto type = func.getFunctionType();

      SmallVector<DictionaryAttr> argAttrs, resAttrs;
      func.getAllArgAttrs(argAttrs);
      func.getAllResultAttrs(resAttrs);

      auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
      funcFunc.setAllArgAttrs(argAttrs);
      funcFunc.setAllResultAttrs(resAttrs);

      auto &funcFuncBody = funcFunc.getBody();
      auto &funcBody = func.getBody();

      IRMapping map;
      funcBody.cloneInto(&funcFuncBody, map);

      for (Block &block : funcFuncBody.getBlocks()) {
        auto term = block.getTerminator();
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
      func.erase();
    });
  }
};

} // namespace
