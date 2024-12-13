import pytest
import torch

import triton
import triton.language as tl


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


@triton.jit
def matmul_kernel_amx(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr,
                      # number of blocks in a group
                      GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_N: tl.constexpr, BLOCKED_A: tl.constexpr,
                      BLOCKED_B: tl.constexpr, TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr):
    pid = tl.program_id(axis=0)
    group_id = pid // (GROUP_SIZE_M * GROUP_SIZE_N)
    groups_n = N // BLOCK_SIZE_N // GROUP_SIZE_N
    group_m = group_id // groups_n
    group_n = group_id % groups_n
    block_id = pid % (GROUP_SIZE_M * GROUP_SIZE_N)
    block_m = group_m * GROUP_SIZE_M + block_id // GROUP_SIZE_N
    block_n = group_n * GROUP_SIZE_N + block_id % GROUP_SIZE_N

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


@pytest.mark.parametrize("M, N, K",
                         [(m, n, k) for m in (128, 256, 512) for n in (128, 256, 512) for k in (128, 256, 512)])
@pytest.mark.parametrize("lhs_dtype, rhs_dtype, res_dtype", [('bfloat16', 'bfloat16', 'float32'),
                                                             ('float16', 'float16', 'float32')])
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K",
                         [(m, n, k) for m in (16, 32) for n in (16, 32) for k in (32, 64)])
@pytest.mark.parametrize("GROUP_SIZE_M, GROUP_SIZE_N", [(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)])
@pytest.mark.parametrize("BLOCKED_A", [False, True])
@pytest.mark.parametrize("BLOCKED_B, TRANSPOSED_B", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("PACKED_B", [False, True])
def test_matmul_amx(M, N, K, lhs_dtype, rhs_dtype, res_dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
                    GROUP_SIZE_N, BLOCKED_A, BLOCKED_B, TRANSPOSED_B, PACKED_B, device):
    assert M % (GROUP_SIZE_M * BLOCK_SIZE_M) == 0, f"M={M}, GROUP_SIZE_M={GROUP_SIZE_M}, BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert N % (GROUP_SIZE_N * BLOCK_SIZE_N) == 0, f"N={N}, GROUP_SIZE_N={GROUP_SIZE_N}, BLOCK_SIZE_N={BLOCK_SIZE_N}"
    assert K % BLOCK_SIZE_K == 0, f"K={K}, BLOCK_SIZE_K={BLOCK_SIZE_K}"
    assert BLOCKED_B or not TRANSPOSED_B

    a = torch.randn((M, K), device=device, dtype=getattr(torch, lhs_dtype))
    b = torch.randn((K, N), device=device, dtype=getattr(torch, rhs_dtype))
    c = torch.zeros((M, N), device=device, dtype=getattr(torch, res_dtype))

    ref = torch.matmul(a.to(c.dtype), b.to(c.dtype))

    if BLOCKED_A:
        ab = torch.empty_like(a)
        prepack_kernel[(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K)](a, ab, M, K, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                               BLOCK_SIZE_N=BLOCK_SIZE_K, BLOCKED_OUTPUT=True,
                                                               TRANSPOSE=False, PACK32=False)
        a = ab

    if BLOCKED_B or PACKED_B:
        bb = torch.empty_like(b)
        prepack_kernel[(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)](b, bb, K, N, BLOCK_SIZE_M=BLOCK_SIZE_K,
                                                               BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCKED_OUTPUT=BLOCKED_B,
                                                               TRANSPOSE=TRANSPOSED_B, PACK32=PACKED_B)
        b = bb

    grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )
    matmul_kernel_amx[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M, GROUP_SIZE_N=GROUP_SIZE_N,  #
        BLOCKED_A=BLOCKED_A, BLOCKED_B=BLOCKED_B,  #
        TRANSPOSED_B=TRANSPOSED_B, PACKED_B=PACKED_B)

    torch.testing.assert_close(c, ref, atol=1e-2, rtol=0)


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


@triton.jit
def block_transpose_combined_kernel(in_a, out_a, in_b, out_b, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                                    GROUP_SIZE_M: tl.constexpr, BLOCKED_A: tl.constexpr, TRANSPOSED_BLOCK_A: tl.constexpr,
                                    BLOCKED_B: tl.constexpr, TRANSPOSED_B: tl.constexpr):
    tl.static_assert(M % BLOCK_SIZE_M == 0)
    tl.static_assert(N % BLOCK_SIZE_N == 0)
    tl.static_assert(BLOCKED_A or not TRANSPOSED_BLOCK_A)
    tl.static_assert(BLOCKED_B or not TRANSPOSED_B)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    in_block_m = first_pid_m + (pid % group_size_m)
    in_block_n = (pid % num_pid_in_group) // group_size_m

    if BLOCKED_A:
        a_out_block_m = in_block_m
        A_OUT_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_K if TRANSPOSED_BLOCK_A else BLOCK_SIZE_M
        A_OUT_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_M if TRANSPOSED_BLOCK_A else BLOCK_SIZE_K
        A_OUT_BLOCKS_M: tl.constexpr = M // BLOCK_SIZE_M
        A_OUT_BLOCKS_K: tl.constexpr = K // BLOCK_SIZE_K
        A_OUT_STRIDE_M: tl.constexpr = A_OUT_BLOCK_SIZE_K
        A_OUT_STRIDE_BLOCK_M: tl.constexpr = BLOCK_SIZE_M * K
        A_OUT_STRIDE_BLOCK_K: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_K
        for in_block_k in tl.range(in_block_n, A_OUT_BLOCKS_K, N // BLOCK_SIZE_N):
            a_out_block_k = in_block_k
            a_in_ptr = tl.make_block_ptr(base=in_a, shape=(M, K), strides=(K, 1),
                                         offsets=(in_block_m * BLOCK_SIZE_M, in_block_k * BLOCK_SIZE_K),
                                         block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
            a_out_ptr = tl.make_block_ptr(
                base=out_a, shape=(A_OUT_BLOCKS_M, A_OUT_BLOCKS_K, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K),
                strides=(A_OUT_STRIDE_BLOCK_M, A_OUT_STRIDE_BLOCK_K, A_OUT_STRIDE_M, 1),
                offsets=(a_out_block_m, a_out_block_k, 0, 0),
                block_shape=(1, 1, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K), order=(3, 2, 1, 0))
            val = tl.load(a_in_ptr)
            if TRANSPOSED_BLOCK_A:
                val = val.T
            val = tl.reshape(val, (1, 1, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K))
            tl.store(a_out_ptr, val)

    if BLOCKED_B:
        B_OUT_BLOCKS_K: tl.constexpr = N // BLOCK_SIZE_N if TRANSPOSED_B else K // BLOCK_SIZE_K
        B_OUT_BLOCKS_N: tl.constexpr = K // BLOCK_SIZE_K if TRANSPOSED_B else N // BLOCK_SIZE_N
        B_OUT_STRIDE_K: tl.constexpr = BLOCK_SIZE_N
        B_OUT_STRIDE_BLOCK_K: tl.constexpr = (K * BLOCK_SIZE_N if TRANSPOSED_B else BLOCK_SIZE_K * N)
        B_OUT_STRIDE_BLOCK_N: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_N
        for in_block_k in tl.range(in_block_m, K // BLOCK_SIZE_K, M // BLOCK_SIZE_M):
            b_out_block_k = in_block_n if TRANSPOSED_B else in_block_k
            b_out_block_n = in_block_k if TRANSPOSED_B else in_block_n
            b_in_ptr = tl.make_block_ptr(base=in_b, shape=(K, N), strides=(N, 1),
                                         offsets=(in_block_k * BLOCK_SIZE_K, in_block_n * BLOCK_SIZE_N),
                                         block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))
            b_out_ptr = tl.make_block_ptr(base=out_b,
                                          shape=(B_OUT_BLOCKS_K, B_OUT_BLOCKS_N, BLOCK_SIZE_K, BLOCK_SIZE_N),
                                          strides=(B_OUT_STRIDE_BLOCK_K, B_OUT_STRIDE_BLOCK_N, B_OUT_STRIDE_K, 1),
                                          offsets=(b_out_block_k, b_out_block_n, 0, 0),
                                          block_shape=(1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N), order=(3, 2, 1, 0))
            val = tl.load(b_in_ptr)
            val = tl.reshape(val, (1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N))
            tl.store(b_out_ptr, val)


@triton.jit
def matmul_kernel_fma(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr,
                      # number of blocks in a group
                      GROUP_SIZE_M: tl.constexpr, BLOCKED_A: tl.constexpr, TRANSPOSED_BLOCK_A: tl.constexpr,
                      BLOCKED_B: tl.constexpr, TRANSPOSED_B: tl.constexpr):
    # TRANSPOSED_BLOCK_A means that each block in A is transposed.
    # It is allowed only for blocked input.
    assert (BLOCKED_A or not TRANSPOSED_BLOCK_A)
    # TRANSPOSED_B means that blocks of B are reordered but blocks
    # itself are not transpoed. It is allowed only for blocked input.
    assert (BLOCKED_B or not TRANSPOSED_B)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    block_m = first_pid_m + (pid % group_size_m)
    block_n = (pid % num_pid_in_group) // group_size_m

    A_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_K if TRANSPOSED_BLOCK_A else BLOCK_SIZE_M
    A_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_M if TRANSPOSED_BLOCK_A else BLOCK_SIZE_K
    A_BLOCKS_M = M // BLOCK_SIZE_M
    A_BLOCKS_K = K // BLOCK_SIZE_K
    a_stride_k = 1
    a_stride_m = A_BLOCK_SIZE_K if BLOCKED_A else K
    a_stride_block_k = A_BLOCK_SIZE_M * A_BLOCK_SIZE_K if BLOCKED_A else A_BLOCK_SIZE_K
    a_stride_block_m = BLOCK_SIZE_M * K

    b_stride_n = 1
    b_stride_k = BLOCK_SIZE_N if BLOCKED_B else N
    if TRANSPOSED_B:
        b_stride_block_n = BLOCK_SIZE_N * K
        b_stride_block_k = BLOCK_SIZE_K * BLOCK_SIZE_N
    else:
        b_stride_block_n = BLOCK_SIZE_K * BLOCK_SIZE_N if BLOCKED_B else BLOCK_SIZE_N
        b_stride_block_k = BLOCK_SIZE_K * N

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(A_BLOCKS_M, A_BLOCKS_K, A_BLOCK_SIZE_M, A_BLOCK_SIZE_K),
                                    strides=(a_stride_block_m, a_stride_block_k, a_stride_m, a_stride_k),
                                    offsets=(block_m, 0, 0, 0), block_shape=(1, 1, A_BLOCK_SIZE_M, A_BLOCK_SIZE_K),
                                    order=(3, 2, 1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr,
                                    shape=(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    strides=(b_stride_block_k, b_stride_block_n, b_stride_k, b_stride_n),
                                    offsets=(0, block_n, 0, 0), block_shape=(1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(3, 2, 1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1),
                                    offsets=(block_m * BLOCK_SIZE_M, block_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr).reshape((A_BLOCK_SIZE_M, A_BLOCK_SIZE_K))
        b = tl.load(b_block_ptr).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N))

        if TRANSPOSED_BLOCK_A:
            a = a.T

        c += tl.dot(a, b, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, 1, 0, 0))
        b_block_ptr = tl.advance(b_block_ptr, (1, 0, 0, 0))

    tl.store(c_block_ptr, c)


@pytest.mark.parametrize("M, N, K",
                         [(m, n, k) for m in (64, 128, 256) for n in (128, 256, 512) for k in (64, 128, 256)])
@pytest.mark.parametrize("lhs_dtype, rhs_dtype, res_dtype", [('float32', 'float32', 'float32')])
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K",
                         [(m, n, k) for m in (8, 16) for n in (16, 32) for k in (4, 8, 16)])
@pytest.mark.parametrize("GROUP_SIZE_M", [4])
@pytest.mark.parametrize("BLOCKED_A, TRANSPOSED_BLOCK_A", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("BLOCKED_B, TRANSPOSED_B", [(False, False), (True, False), (True, True)])
def test_matmul_fma(M, N, K, lhs_dtype, rhs_dtype, res_dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
                    BLOCKED_A, TRANSPOSED_BLOCK_A, BLOCKED_B, TRANSPOSED_B, device):
    assert M % (GROUP_SIZE_M * BLOCK_SIZE_M) == 0, f"M={M}, GROUP_SIZE_M={GROUP_SIZE_M}, BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert K % BLOCK_SIZE_K == 0, f"K={K}, BLOCK_SIZE_K={BLOCK_SIZE_K}"
    assert BLOCKED_B or not TRANSPOSED_B

    a = torch.randn((M, K), device=device, dtype=getattr(torch, lhs_dtype))
    b = torch.randn((K, N), device=device, dtype=getattr(torch, rhs_dtype))
    c = torch.zeros((M, N), device=device, dtype=getattr(torch, res_dtype))

    ref = torch.matmul(a.to(c.dtype), b.to(c.dtype))

    #if BLOCKED_A:
    #    ab = torch.empty_like(a)
    #    block_transpose_kernel[(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K)](a, ab, M, K, BLOCK_SIZE_M=BLOCK_SIZE_M,
    #                                                                   BLOCK_SIZE_N=BLOCK_SIZE_K, BLOCKED_OUTPUT=True,
    #                                                                   TRANSPOSE_OUTER=False, TRANSPOSE_INNER=TRANSPOSED_BLOCK_A)
    #    a = ab

    #if BLOCKED_B:
    #    bb = torch.empty_like(b)
    #    block_transpose_kernel[(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)](b, bb, K, N, BLOCK_SIZE_M=BLOCK_SIZE_K,
    #                                                                   BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCKED_OUTPUT=True,
    #                                                                   TRANSPOSE_OUTER=TRANSPOSED_B, TRANSPOSE_INNER=False)
    #    b = bb

    grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )

    if BLOCKED_A or BLOCKED_B:
        ab = torch.empty((M * K + (M // BLOCK_SIZE_M) * (K // BLOCK_SIZE_K) * 64), device=device,
                         dtype=getattr(torch, lhs_dtype))
        bb = torch.empty((K * N + (K // BLOCK_SIZE_K) * (N // BLOCK_SIZE_N) * 64), device=device,
                         dtype=getattr(torch, lhs_dtype))
        block_transpose_combined_kernel[grid](
            a, ab, b, bb,  #
            M, N, K,  #
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            BLOCKED_A=BLOCKED_A, TRANSPOSED_BLOCK_A=TRANSPOSED_BLOCK_A,  #
            BLOCKED_B=BLOCKED_B, TRANSPOSED_B=TRANSPOSED_B)
        if BLOCKED_A:
            a = ab
        if BLOCKED_B:
            b = bb

    matmul_kernel_fma[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        BLOCKED_A=BLOCKED_A, TRANSPOSED_BLOCK_A=TRANSPOSED_BLOCK_A,  #
        BLOCKED_B=BLOCKED_B, TRANSPOSED_B=TRANSPOSED_B)

    torch.testing.assert_close(c, ref, atol=1e-2, rtol=0)
