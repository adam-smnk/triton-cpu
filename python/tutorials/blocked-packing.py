"""
Blocked packing
=====================
Blocked packing can be used for matmul input to improve memory access patterns
and significantly increase matmul performance.

"""

import torch
import pandas

import triton
import triton.language as tl

BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 64

# %%
# Kernel
# ------


@triton.jit
def prepack_kernel(in_p, out_p, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                   BLOCK_SIZE_N: tl.constexpr, BLOCKED_OUTPUT: tl.constexpr, TRANSPOSE: tl.constexpr,
                   PACK32: tl.constexpr):
    B_PACK_SCALE: tl.constexpr = 32 // in_p.type.element_ty.primitive_bitwidth if PACK32 else 1
    tl.static_assert(B_PACK_SCALE >= 1 and B_PACK_SCALE <= 4)
    tl.static_assert(M % BLOCK_SIZE_M == 0)
    tl.static_assert(N % BLOCK_SIZE_N == 0)
    tl.static_assert(BLOCK_SIZE_M % B_PACK_SCALE == 0)
    tl.static_assert(BLOCKED_OUTPUT or not TRANSPOSE)
    tl.static_assert(BLOCKED_OUTPUT or PACK32)
    in_block_m = tl.program_id(0)
    in_block_n = tl.program_id(1)
    out_block_m = in_block_n if TRANSPOSE else in_block_m
    out_block_n = in_block_m if TRANSPOSE else in_block_n
    BLOCK_IN_OFFS = in_block_m * BLOCK_SIZE_M * N + in_block_n * BLOCK_SIZE_N
    OUT_STRIDE_M: tl.constexpr = BLOCK_SIZE_N * B_PACK_SCALE if BLOCKED_OUTPUT else N * B_PACK_SCALE
    OUT_STRIDE_BLOCK_N: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N if BLOCKED_OUTPUT else BLOCK_SIZE_N * B_PACK_SCALE
    OUT_STRIDE_BLOCK_M: tl.constexpr = M * BLOCK_SIZE_N if TRANSPOSE else BLOCK_SIZE_M * N
    BLOCK_OUT_OFFS = out_block_m * OUT_STRIDE_BLOCK_M + out_block_n * OUT_STRIDE_BLOCK_N
    for i in tl.range(0, BLOCK_SIZE_M // B_PACK_SCALE):
        row1 = tl.load(in_p + BLOCK_IN_OFFS + N * i * B_PACK_SCALE + tl.arange(0, BLOCK_SIZE_N))
        if B_PACK_SCALE > 1:
            row2 = tl.load(in_p + BLOCK_IN_OFFS + N * (i * B_PACK_SCALE + 1) + tl.arange(0, BLOCK_SIZE_N))
            if B_PACK_SCALE > 2:
                row3 = tl.load(in_p + BLOCK_IN_OFFS + N * (i * B_PACK_SCALE + 2) + tl.arange(0, BLOCK_SIZE_N))
                row4 = tl.load(in_p + BLOCK_IN_OFFS + N * (i * B_PACK_SCALE + 3) + tl.arange(0, BLOCK_SIZE_N))
                row1 = tl.ravel(tl.join(row1, row3))
                row2 = tl.ravel(tl.join(row2, row4))
            row1 = tl.ravel(tl.join(row1, row2))
        tl.store(out_p + BLOCK_OUT_OFFS + OUT_STRIDE_M * i + tl.arange(0, BLOCK_SIZE_N * B_PACK_SCALE), row1)


def prepack(a: torch.Tensor, res: torch.Tensor, TRANSPOSE, PACK32, num_threads=0):
    M, N = a.shape
    #TODO: Currently masked load is not supported yet.
    assert (M % BLOCK_SIZE_M
            == 0) and (N % BLOCK_SIZE_N
                       == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    prepack_kernel[(M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)](a, res, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                           BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCKED_OUTPUT=True,
                                                           TRANSPOSE=TRANSPOSE, PACK32=PACK32, num_threads=num_threads)
    return res


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation.
torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

M = 256
N = 1024
a = torch.randn((M, N), device='cpu', dtype=torch.bfloat16)
res = torch.empty((N // BLOCK_SIZE_N, M // BLOCK_SIZE_M, BLOCK_SIZE_M // 2, BLOCK_SIZE_N * 2), dtype=torch.bfloat16,
                  device='cpu')
prepack(a, res, True, True)
for m in range(0, M):
    for n in range(0, N):
        block_m = n // BLOCK_SIZE_N
        block_n = m // BLOCK_SIZE_M
        new_m = m % BLOCK_SIZE_M // 2
        new_n = n % BLOCK_SIZE_N * 2 + m % 2
        assert a[m, n] == res[
            block_m, block_n, new_m,
            new_n], f"Mismatch for ({m}, {n}) mapped to ({block_m}, {block_n}, {new_m}, {new_n}) ({a[m, n]} != {res[block_m, block_n, new_m, new_n]}) BLOCK_SIZE_M={BLOCK_SIZE_M} BLOCK_SIZE_N={BLOCK_SIZE_N}"
print("✅ Triton result is OK")

# %%
# Benchmark options
# -----------------

PACK_OPTS = ['blocked', 'blocked-pack32', 'blocked-transposed', 'blocked-transposed-pack32', 'pack32']
LINE_VALS = [f'triton-cpu-{opt}{prefix}' for prefix in ['-single', ''] for opt in PACK_OPTS]
LINE_NAMES = LINE_VALS
LINE_STYLES = [('blue', '--')] * len(LINE_VALS)

# %%
# Benchmark
# ---------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 21)],  # Different possible values for `x_name`
        #x_vals=[128 * i for i in range(2, 21, 4)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'pack-performance-bf16 (BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N})',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, provider):

    device = 'cpu'
    a = torch.randn((M, N), device=device, dtype=torch.bfloat16)
    res = torch.zeros_like(a)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: prepack(a, res, 'transposed' in provider, 'pack32' in provider, num_threads=int('single' in provider)),
        quantiles=quantiles, measure_time_with_hooks=True)
    perf = lambda ms: 2 * M * N * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
df = benchmark.run(print_data=True, show_plots=True, return_df=True)

dfs = []
for single in [True, False]:
    for opt in PACK_OPTS:
        col_name = f"triton-cpu-{opt}{'-single' if single else ''}"
        opt_df = pandas.DataFrame(
            data={
                'N': df['N'], 'Threading': 'Single Thread' if single else 'Multiple Threads', 'Encoding': opt, 'GB/s':
                df[col_name]
            })
        dfs.append(opt_df)
df = pandas.concat(dfs)
df.to_csv('triton-cpu-pack.csv', index=False)
