"""
Matrix Multiplication
=====================
In this tutorial, matmul on CPU with different input layouts is tested.

This tutorial is optimized for AMX-enabled CPUs.

"""

# %%
# Kernels
# -------

import torch
import pandas

import triton
import triton.language as tl

BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 8
GROUP_SIZE_N = 4


# This kernel is used for blocked and/or VNNI encoding of input tensors
# for matmul.
#
# Blocked encoding is used to transform 2D tensor [M, N] into 4D tensor
# [M / BLOCK_SIZE_M, N / BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_N].
# This makes following access to blocks in matmul more efficient because
# each block is placed into a contiguous memory fragment and is likely
# to fit a single memory page.
#
# If TRANSPOSE is set to True then head dimensions of the output
# tensor are transposed. It provides contiguos placement for a column
# of blocks making it more afficient. It should be used for RHS of
# the matmul only. Only available for blocked encoding.
#
# PACK32 option is used to provide VNNI encoding within each block
# (or the whole tensor is blocked encoding is not requested). In this
# encoding, rows are interleaved so that elements of the same input
# column are packed into 32-bit groups within a row. E.g. BF16 input
#  [[ 1,  2,  3,  4],
#   [ 5,  6,  7,  8],
#   [ 9, 10, 11, 12],
#   [13, 14, 15, 16]]
# is transformed into
#  [[ 1,  5,  2,  6,  3,  7,  4,  8],
#   [ 9, 13, 10, 14, 11, 15, 12, 16]]
# With such encoding, data can be loaded directly to AMX registers
# without additional preprations. It should be used for RHS of the
# matmul only when AMX is available.
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


# Matmul kernel that computes a single output block [BLOCK_SIZE_M, BLOCK_SIZE_N]. LHS can be in the
# rowmajor or blocked encoding. RHS can be in rowmajor, blocked, blocked and transposed, packed, or
# blocked transposed packed encoding.
#
# To cover all input layouts, we use 4D block pointers that address a single input block
# [1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N] or [1, 1, BLOCK_SIZE_M // SCALE, BLOCK_SIZE_N * SCALE] for
# VNNI packed input. Depending on actual input layout, we choose strides for these block pointers
# appropriately to keep navigation bentween blocks similar for all input encodings.
#
# E.g. for rowmajor LHS we use BLOCK_SIZE_K stride to move to the next block over K axis, but
# for blocked encoding we use BLOCK_SIZE_M * BLOCK_SIZE_K stride. In both cases we then can
# advance using the same (0, 1, 0, 0) offset in the loop.
#
# Reshape is used to remove the heading (1, 1) dimensions, but CPU backend folds it with the load
# operation and it doesn't prevent direct AMX tile loads from the input memory (when encoding
# allows that).
@triton.jit
def matmul_kernel_amx(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr,
                      # number of blocks in a group
                      GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_N: tl.constexpr, BLOCKED_A: tl.constexpr,
                      BLOCKED_B: tl.constexpr, TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr):
    #pid = tl.program_id(axis=0)
    #group_id = pid // (GROUP_SIZE_M * GROUP_SIZE_N)
    #groups_n = N // BLOCK_SIZE_N // GROUP_SIZE_N
    #group_m = group_id // groups_n
    #group_n = group_id % groups_n
    #block_id = pid % (GROUP_SIZE_M * GROUP_SIZE_N)
    #block_m = group_m * GROUP_SIZE_M + block_id // GROUP_SIZE_N
    #block_n = group_n * GROUP_SIZE_N + block_id % GROUP_SIZE_N
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    block_m = first_pid_m + (pid % group_size_m)
    block_n = (pid % num_pid_in_group) // group_size_m

    a_stride_k = 1
    a_stride_m = BLOCK_SIZE_K if BLOCKED_A else K
    a_stride_block_k = BLOCK_SIZE_M * BLOCK_SIZE_K if BLOCKED_A else BLOCK_SIZE_K
    a_stride_block_m = BLOCK_SIZE_M * K

    B_PACK_SCALE: tl.constexpr = 32 // b_ptr.type.element_ty.primitive_bitwidth if PACKED_B else 1
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // B_PACK_SCALE if PACKED_B else BLOCK_SIZE_K
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N * B_PACK_SCALE if PACKED_B else BLOCK_SIZE_N
    assert BLOCKED_B or not TRANSPOSED_B
    b_stride_n = 1
    b_stride_k = PACKED_BLOCK_SIZE_N if BLOCKED_B else N * B_PACK_SCALE
    if TRANSPOSED_B:
        b_stride_block_n = BLOCK_SIZE_N * K
        b_stride_block_k = BLOCK_SIZE_K * BLOCK_SIZE_N
    else:
        b_stride_block_n = BLOCK_SIZE_K * BLOCK_SIZE_N if BLOCKED_B else PACKED_BLOCK_SIZE_N
        b_stride_block_k = BLOCK_SIZE_K * N

    a_block_ptr = tl.make_block_ptr(base=a_ptr,
                                    shape=(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    strides=(a_stride_block_m, a_stride_block_k, a_stride_m, a_stride_k),
                                    offsets=(block_m, 0, 0, 0), block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(3, 2, 1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N),
        strides=(b_stride_block_k, b_stride_block_n, b_stride_k, b_stride_n), offsets=(0, block_n, 0, 0),
        block_shape=(1, 1, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N), order=(3, 2, 1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1),
                                    offsets=(block_m * BLOCK_SIZE_M, block_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = tl.load(b_block_ptr).reshape((PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N))

        c += tl.dot(a, b, out_dtype=tl.float32, rhs_encoding="row_major_interleaved" if PACKED_B else "row_major")

        a_block_ptr = tl.advance(a_block_ptr, (0, 1, 0, 0))
        b_block_ptr = tl.advance(b_block_ptr, (1, 0, 0, 0))

    tl.store(c_block_ptr, c)


def matmul_amx(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, ab: torch.Tensor, bb: torch.Tensor, PREPACKED,
               BLOCKED_A, BLOCKED_B, TRANSPOSED_B, PACKED_B, num_threads=0):
    M, K = a.shape
    N = c.shape[1]
    #TODO: Currently masked load is not supported yet.
    #assert (M % (BLOCK_SIZE_M * GROUP_SIZE_M) == 0) and (N % (BLOCK_SIZE_N * GROUP_SIZE_N) == 0) and (
    #    K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    assert (M % BLOCK_SIZE_M == 0) and (N % BLOCK_SIZE_N == 0) and (
        K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    if BLOCKED_A and not PREPACKED:
        prepack_kernel[(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K)](a, ab, M, K, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                               BLOCK_SIZE_N=BLOCK_SIZE_K, BLOCKED_OUTPUT=True,
                                                               TRANSPOSE=False, PACK32=False, num_threads=num_threads)
        a = ab
    if (BLOCKED_B or PACKED_B) and not PREPACKED:
        prepack_kernel[(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)](b, bb, K, N, BLOCK_SIZE_M=BLOCK_SIZE_K,
                                                               BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCKED_OUTPUT=BLOCKED_B,
                                                               TRANSPOSE=TRANSPOSED_B, PACK32=PACKED_B,
                                                               num_threads=num_threads)
        b = bb
    # 1D launch kernel where each block gets its own program.
    grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )
    matmul_kernel_amx[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M, GROUP_SIZE_N=GROUP_SIZE_N,  #
        BLOCKED_A=BLOCKED_A, BLOCKED_B=BLOCKED_B,  #
        TRANSPOSED_B=TRANSPOSED_B, PACKED_B=PACKED_B, num_threads=num_threads)
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation.
torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

a = torch.randn((512, 512), device='cpu', dtype=torch.bfloat16)
b = torch.randn((512, 512), device='cpu', dtype=torch.bfloat16)
c = torch.empty((512, 512), device='cpu', dtype=torch.float32)
torch_output = torch.matmul(a.to(torch.float32), b.to(torch.float32))
rtol = 0
a_tmp = torch.zeros_like(a)
b_tmp = torch.zeros_like(b)
triton_output = matmul_amx(a, b, c, a_tmp, b_tmp, False, False, False, False, False)
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU and TorchCPU match")
else:
    print("❌ TritonCPU and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
    assert False
triton_output = matmul_amx(a, b, c, a_tmp, b_tmp, False, True, True, True, True)
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU pre-packed and TorchCPU match")
else:
    print("❌ TritonCPU pre-packed and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
    assert False

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of Pytorch. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.


def encode_provider(blocked_a, blocked_b, transposed_b, packed_b, prepack, single_thread, dtype):
    assert dtype == 'float32' or dtype == 'bfloat16' or dtype == 'float16'
    return f"triton-cpu{'-ba' if blocked_a else ''}{'-bb' if blocked_b else ''}{'-tb' if transposed_b else ''}{'-vnni' if packed_b else ''}{'-pre' if prepack else ''}{'-st' if single_thread else ''}-{dtype}"


def decode_provider(provider):
    if '-bfloat16' in provider:
        dtype = torch.bfloat16
    if '-float16' in provider:
        dtype = torch.float16
    elif '-float32' in provider:
        dtype = torch.float32
    if 'triton-cpu' in provider:
        backend = 'triton-cpu'
    elif 'torch-cpu-native' in provider:
        backend = 'torch-cpu-native'
    elif 'torch-cpu-compile' in provider:
        backend = 'torch-cpu-compile'
    return backend, '-ba' in provider, '-bb' in provider, '-tb' in provider, '-vnni' in provider, '-pre' in provider, '-st' in provider, dtype


BLOCKED_A_BLOCKED_TRANSPOSED_PACKED_B_OPTS = [(True, True, True, True), (False, True, True, True),
                                              (False, False, False, False)]
PREPACK_OPTS = [False, True]
SINGLE_THREAD_OPTS = [True]
DTYPE_OPTS = ['bfloat16']
LINE_VALS = [
    encode_provider(blocked_a, blocked_b, transposed_b, packed_b, prepack, single_thread, dtype)
    for blocked_a, blocked_b, transposed_b, packed_b in BLOCKED_A_BLOCKED_TRANSPOSED_PACKED_B_OPTS
    for prepack in PREPACK_OPTS
    for single_thread in SINGLE_THREAD_OPTS
    for dtype in DTYPE_OPTS
    if (not packed_b or dtype != 'float32') and (blocked_a or blocked_b or packed_b or not prepack)
] + [f'torch-cpu-native-{dtype}' for dtype in DTYPE_OPTS]
LINE_NAMES = LINE_VALS
LINE_STYLES = None  #[('blue', '--')] * len(LINE_VALS)

default_num_threads = torch.get_num_threads()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 21)],  # Different possible values for `x_name`
        #x_vals=[4096, 8192],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GFLOPS',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'matmul-performance-bf16 (BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M}), GROUP_SIZE_N={GROUP_SIZE_N})',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, K, provider):

    device = 'cpu' if 'cpu' in provider else 'cuda'
    backend, blocked_a, blocked_b, transposed_b, packed_b, prepack, single_thread, dtype = decode_provider(provider)
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)

    if single_thread:
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(default_num_threads)

    if backend == 'triton-cpu':
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
        a_tmp = torch.zeros_like(a)
        b_tmp = torch.zeros_like(b)
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
        if prepack and blocked_a:
            ab = torch.empty_like(a)
            prepack_kernel[(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K)](a, ab, M, K, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                                   BLOCK_SIZE_N=BLOCK_SIZE_K, BLOCKED_OUTPUT=True,
                                                                   TRANSPOSE=False, PACK32=False)
            a = ab
        if prepack and (blocked_b or packed_b):
            bb = torch.empty_like(b)
            prepack_kernel[(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)](b, bb, K, N, BLOCK_SIZE_M=BLOCK_SIZE_K,
                                                                   BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCKED_OUTPUT=blocked_b,
                                                                   TRANSPOSE=transposed_b, PACK32=packed_b)
            b = bb
    else:
        c = torch.zeros((M, N), device=a.device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if backend == 'torch-cpu-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b, out=c), quantiles=quantiles)
    elif backend == 'torch-cpu-compile':
        compiled = torch.compile(torch.matmul)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled(a, b, out=c), quantiles=quantiles)
    elif backend == 'triton-cpu':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_amx(a, b, c, a_tmp, b_tmp, prepack, blocked_a, blocked_b, transposed_b, packed_b, num_threads
                               =int(single_thread)), quantiles=quantiles, measure_time_with_hooks=True)
    perf = lambda ms: 2 * M * N * K * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
df = benchmark.run(print_data=True, show_plots=True, return_df=True)

# %%
# Write measurement results into CSV file
dfs = []
for provider in LINE_VALS:
    backend, ba, bb, tb, pack32, prepack, single, dtype = decode_provider(provider)
    dtype = dtype.__reduce__()
    encoding = f"{'-BA' if ba else ''}{'-BB' if bb else ''}{'-TB' if tb else ''}{'-vnni' if pack32 else ''}"
    if backend == 'triton-cpu':
        encoding = encoding[1:] if len(encoding) > 0 else 'None'
    elif backend == 'torch-cpu-native':
        encoding = 'Torch Native'
    elif backend == 'torch-cpu-compile':
        encoding = 'Torch Inductor'
    opt_df = pandas.DataFrame(
        data={
            'N': df['N'], 'Threading': 'Single Thread' if single else 'Multiple Threads', 'Option': encoding,
            'Prepacked': 'Yes' if prepack else 'No', 'Type': dtype, 'GFLOP/s': df[provider]
        })
    dfs.append(opt_df)
df = pandas.concat(dfs)
df.to_csv('triton-cpu-matmul.csv', index=False)
