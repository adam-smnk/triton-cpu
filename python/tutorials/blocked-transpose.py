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

BLOCK_SIZE_M = 8
BLOCK_SIZE_N = 16

# %%
# Kernel
# ------


@triton.jit
def block_transpose_kernel(in_p, out_p, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                           BLOCK_SIZE_N: tl.constexpr, BLOCKED_OUTPUT: tl.constexpr, TRANSPOSE_OUTER: tl.constexpr,
                           TRANSPOSE_INNER: tl.constexpr):
    tl.static_assert(M % BLOCK_SIZE_M == 0)
    tl.static_assert(N % BLOCK_SIZE_N == 0)
    tl.static_assert(BLOCKED_OUTPUT or not TRANSPOSE_OUTER)
    tl.static_assert(BLOCKED_OUTPUT or TRANSPOSE_INNER)
    in_block_m = tl.program_id(0)
    in_block_n = tl.program_id(1)
    out_block_m = in_block_n if TRANSPOSE_OUTER else in_block_m
    out_block_n = in_block_m if TRANSPOSE_OUTER else in_block_n
    OUT_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_N if TRANSPOSE_INNER else BLOCK_SIZE_M
    OUT_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_M if TRANSPOSE_INNER else BLOCK_SIZE_N
    OUT_BLOCKS_M = N // BLOCK_SIZE_N if TRANSPOSE_OUTER else M // BLOCK_SIZE_M
    OUT_BLOCKS_N = M // BLOCK_SIZE_M if TRANSPOSE_OUTER else N // BLOCK_SIZE_N
    OUT_STRIDE_M: tl.constexpr = OUT_BLOCK_SIZE_N if BLOCKED_OUTPUT else N
    OUT_STRIDE_BLOCK_N: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N if BLOCKED_OUTPUT else OUT_BLOCK_SIZE_N
    OUT_STRIDE_BLOCK_M: tl.constexpr = M * BLOCK_SIZE_N if TRANSPOSE_OUTER else BLOCK_SIZE_M * N

    in_ptr = tl.make_block_ptr(base=in_p, shape=(M, N), strides=(N, 1),
                               offsets=(in_block_m * BLOCK_SIZE_M, in_block_n * BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    out_ptr = tl.make_block_ptr(base=out_p, shape=(OUT_BLOCKS_M, OUT_BLOCKS_N, OUT_BLOCK_SIZE_M, OUT_BLOCK_SIZE_N),
                                strides=(OUT_STRIDE_BLOCK_M, OUT_STRIDE_BLOCK_N, OUT_STRIDE_M, 1),
                                offsets=(out_block_m, out_block_n, 0, 0),
                                block_shape=(1, 1, OUT_BLOCK_SIZE_M, OUT_BLOCK_SIZE_N), order=(3, 2, 1, 0))
    val = tl.load(in_ptr)
    if TRANSPOSE_INNER:
        val = val.T
    val = tl.reshape(val, (1, 1, OUT_BLOCK_SIZE_M, OUT_BLOCK_SIZE_N))
    tl.store(out_ptr, val)


def block_transpose(a: torch.Tensor, res: torch.Tensor, TRANSPOSE_OUTER, TRANSPOSE_INNER, num_threads=0):
    M, N = a.shape
    #TODO: Currently masked load is not supported yet.
    assert (M % BLOCK_SIZE_M
            == 0) and (N % BLOCK_SIZE_N
                       == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    block_transpose_kernel[(M // BLOCK_SIZE_M,
                            N // BLOCK_SIZE_N)](a, res, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                BLOCKED_OUTPUT=True, TRANSPOSE_OUTER=TRANSPOSE_OUTER,
                                                TRANSPOSE_INNER=TRANSPOSE_INNER, num_threads=num_threads)
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
a = torch.randn((M, N), device='cpu', dtype=torch.float32)
res = torch.empty((N // BLOCK_SIZE_N, M // BLOCK_SIZE_M, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device='cpu')
block_transpose(a, res, True, False)
for m in range(0, M):
    for n in range(0, N):
        block_m = n // BLOCK_SIZE_N
        block_n = m // BLOCK_SIZE_M
        new_m = m % BLOCK_SIZE_M
        new_n = n % BLOCK_SIZE_N
        assert a[m, n] == res[
            block_m, block_n, new_m,
            new_n], f"Mismatch for ({m}, {n}) mapped to ({block_m}, {block_n}, {new_m}, {new_n}) ({a[m, n]} != {res[block_m, block_n, new_m, new_n]}) BLOCK_SIZE_M={BLOCK_SIZE_M} BLOCK_SIZE_N={BLOCK_SIZE_N}"
print("✅ Triton result is OK")
res = torch.empty((M // BLOCK_SIZE_M, N // BLOCK_SIZE_N, BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=torch.float32, device='cpu')
block_transpose(a, res, False, True)
for m in range(0, M):
    for n in range(0, N):
        block_m = m // BLOCK_SIZE_M
        block_n = n // BLOCK_SIZE_N
        new_m = n % BLOCK_SIZE_N
        new_n = m % BLOCK_SIZE_M
        assert a[m, n] == res[
            block_m, block_n, new_m,
            new_n], f"Mismatch for ({m}, {n}) mapped to ({block_m}, {block_n}, {new_m}, {new_n}) ({a[m, n]} != {res[block_m, block_n, new_m, new_n]}) BLOCK_SIZE_M={BLOCK_SIZE_M} BLOCK_SIZE_N={BLOCK_SIZE_N}"
print("✅ Triton result is OK")

# %%
# Benchmark options
# -----------------

#SINGLE_THREAD_OPTS = ['-single', '']
SINGLE_THREAD_OPTS = ['']
PACK_OPTS = ['blocked', 'blocked-transposeinner', 'blocked-transposeouter']
LINE_VALS = [f'triton-cpu-{opt}{prefix}' for prefix in SINGLE_THREAD_OPTS for opt in PACK_OPTS]
LINE_NAMES = LINE_VALS
LINE_STYLES = None

# %%
# Benchmark
# ---------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],  # Argument names to use as an x-axis for the plot
        #x_vals=[128 * i for i in range(2, 21)],  # Different possible values for `x_name`
        #x_vals=[128 * i for i in range(16, 21)],  # Different possible values for `x_name`
        x_vals=[i for i in [4096, 8192]],  # Different possible values for `x_name`
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
    a = torch.randn((M, N), device=device, dtype=torch.float32)
    res = torch.zeros_like(a)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: block_transpose(a, res, 'transposeouter' in provider, 'transposeinner' in provider, num_threads=int(
            'single' in provider)), quantiles=quantiles, measure_time_with_hooks=True, rep=500)
    perf = lambda ms: 4 * M * N * 1e-9 / (ms * 1e-3)
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
