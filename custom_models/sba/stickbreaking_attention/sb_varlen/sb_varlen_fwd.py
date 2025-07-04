import torch
import triton
import triton.language as tl

from ..utils import ALLOW_TF32, inv_log2
from .softplus import softplus
from ..utils import custom_op


@triton.jit
def load_kv(K_blk_ptrs, V_blk_ptrs, N_mask, NO_N_MASK, D_mask, NO_D_MASK: tl.constexpr):
    if NO_D_MASK:
        if NO_N_MASK:
            k = tl.load(K_blk_ptrs)
            v = tl.load(V_blk_ptrs)
        else:
            k = tl.load(K_blk_ptrs, mask=N_mask[:, None])
            v = tl.load(V_blk_ptrs, mask=N_mask[:, None])
    else:
        mask = N_mask[:, None] & D_mask[None, :]
        k = tl.load(K_blk_ptrs, mask=mask)
        v = tl.load(V_blk_ptrs, mask=mask)
    return k, v


@triton.jit
def compute_block(
    q,
    k,
    qk_scale,
    neg_log_acc,
    M_blk_idxs,
    N_blk_idxs,
    cm,
    on_band: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    backward: tl.constexpr,
    attend_current: tl.constexpr = False,
    use_cumsum: tl.constexpr = False,
    is_compiling: tl.constexpr = False,
):
    qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * qk_scale

    # log_om_beta (one minus beta) : log(1 - \beta)
    log_om_beta = -softplus(qk, is_compiling=is_compiling)

    if on_band:  # diagonal
        if attend_current:
            block_mask = M_blk_idxs[:, None] >= N_blk_idxs[None, :]
        else:
            block_mask = M_blk_idxs[:, None] > N_blk_idxs[None, :]
        log_om_beta = tl.where(block_mask, log_om_beta, 0.0)
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]

        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm,
                           acc=log_p, allow_tf32=ALLOW_TF32)

        p = tl.math.exp2(log_p)
        p = tl.where(block_mask, p, 0.0)
    else:
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]
        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm,
                           acc=log_p, allow_tf32=ALLOW_TF32)

        p = tl.math.exp2(log_p)
    if not backward:
        neg_log_acc += tl.sum(log_om_beta, axis=1)
    return p, log_om_beta, neg_log_acc


@triton.jit
def _forward_one_row(
    seq_block_id,
    seq_length,
    qk_scale,
    M_range,
    N_range,
    D_range,
    D_mask,
    cm,
    Q_head_seq_ptr,
    stride_qm,
    stride_qd: tl.constexpr,
    K_head_seq_ptr,
    stride_kn,
    stride_kd: tl.constexpr,
    V_head_seq_ptr,
    stride_vn,
    stride_vd: tl.constexpr,
    O_head_seq_ptr,
    stride_om,
    stride_od: tl.constexpr,
    R_head_seq_ptr,
    stride_rm,
    A_head_seq_ptr,
    stride_am,
    W_head_seq_ptr,
    stride_wm,
    stride_wn,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    return_attention: tl.constexpr = False,
    is_compiling: tl.constexpr = False,
    use_cumsum: tl.constexpr = False,
    attend_current: tl.constexpr = False,
):
    # Loading thread information
    block_start_offset = BLOCK_M * seq_block_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    NO_M_MASK = (block_start_offset + BLOCK_M - 1) < seq_length

    # BLOCK_M must be a multiple of BLOCK_N
    N_blk_idxs_start = block_start_offset + BLOCK_M
    N_blk_idxs = N_blk_idxs_start + N_range

    # Init pointers
    Q_blk_ptrs = Q_head_seq_ptr + \
        (stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :])
    K_blk_ptrs = K_head_seq_ptr + \
        (stride_kn * N_blk_idxs[:, None] + stride_kd * D_range[None, :])
    V_blk_ptrs = V_head_seq_ptr + \
        (stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :])
    O_blk_ptrs = O_head_seq_ptr + \
        (stride_om * M_blk_idxs[:, None] + stride_od * D_range[None, :])
    R_blk_ptrs = R_head_seq_ptr + stride_rm * M_blk_idxs
    A_blk_ptrs = A_head_seq_ptr + stride_am * M_blk_idxs

    # --- Load band vectors ---
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.0)
    else:
        q = tl.load(
            Q_blk_ptrs, mask=M_mask[:, None] & D_mask[None, :], other=0.0)

    iters = N_blk_idxs_start // BLOCK_N
    neg_log_acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=acc_dtype)
    # --- End band vectors ---

    # Iterate only up to start of sequence
    for i in range(iters):
        N_blk_idxs -= BLOCK_N
        N_blk_idxs_start -= BLOCK_N
        K_blk_ptrs -= BLOCK_N * stride_kn
        V_blk_ptrs -= BLOCK_N * stride_vn

        N_mask = N_blk_idxs < seq_length
        k, v = load_kv(
            K_blk_ptrs,
            V_blk_ptrs,
            N_mask=N_mask,
            NO_N_MASK=N_blk_idxs_start + BLOCK_N - 1 < seq_length,
            D_mask=D_mask,
            NO_D_MASK=NO_D_MASK,
        )
        on_band = i < BLOCK_M // BLOCK_N
        p, _, neg_log_acc = compute_block(
            q,
            k,
            qk_scale,
            neg_log_acc,
            M_blk_idxs,
            N_blk_idxs,
            cm,
            on_band,
            ALLOW_TF32,
            attend_current=attend_current,
            backward=False,
            is_compiling=is_compiling,
            use_cumsum=use_cumsum,
        )
        # Store intermediate values
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=ALLOW_TF32)
        if return_attention:  # TODO write returns_attention_weight
            tl.store(
                W_head_seq_ptr + stride_wm *
                M_blk_idxs[:, None] + stride_wn * N_blk_idxs[None, :],
                p,
                mask=(M_blk_idxs < seq_length)[:, None] & (
                    N_blk_idxs < seq_length)[None, :],
            )
    if NO_M_MASK:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc))
        tl.store(A_blk_ptrs, neg_log_acc.to(A_head_seq_ptr.type.element_ty))
    else:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc), mask=M_mask)
        tl.store(A_blk_ptrs, neg_log_acc.to(
            A_head_seq_ptr.type.element_ty), mask=M_mask)
    if NO_D_MASK:
        tl.store(O_blk_ptrs, acc.to(
            O_head_seq_ptr.type.element_ty), mask=M_mask[:, None])
    else:
        tl.store(O_blk_ptrs, acc.to(O_head_seq_ptr.type.element_ty),
                 mask=M_mask[:, None] & D_mask[None, :])


def get_configs():
    return [triton.Config({"BLOCK_M": mb, "BLOCK_N": nb}, num_stages=s, num_warps=w)
            for mb in [64, 128]
            for nb in [16, 32, 64]
            for s in [4, 2, 3, 5, 6, 7, 8]
            for w in [4, 2]]


@triton.autotune(configs=get_configs(), key=["head_size"])
@triton.jit
def _forward(
    Q_ptr,
    stride_qh: tl.constexpr,
    stride_qm,
    stride_qd: tl.constexpr,
    K_ptr,
    stride_kh: tl.constexpr,
    stride_kn,
    stride_kd: tl.constexpr,
    V_ptr,
    stride_vh: tl.constexpr,
    stride_vn,
    stride_vd: tl.constexpr,
    O_ptr,
    stride_oh: tl.constexpr,
    stride_om,
    stride_od: tl.constexpr,
    R_ptr,
    stride_rh,
    stride_rm: tl.constexpr,
    A_ptr,
    stride_ah,
    stride_am: tl.constexpr,
    W_ptr,
    stride_wh,
    stride_wm,
    stride_wn,
    CSL_ptr,
    logit_scale: tl.constexpr,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    return_attention: tl.constexpr = False,
    use_cumsum: tl.constexpr = False,
    attend_current: tl.constexpr = False
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    seq_id = tl.program_id(0)
    fhead_id = tl.program_id(1)
    seq_alloc_prog_id = tl.program_id(2)
    num_seq_alloc_progs = tl.num_programs(2)
    if seq_id == 0:
        seq_start_offset = 0
    else:
        seq_start_offset = tl.load(CSL_ptr + seq_id - 1).to(tl.int32)
    seq_end_offset = tl.load(CSL_ptr + seq_id).to(tl.int32)
    seq_length = seq_end_offset - seq_start_offset
    num_seq_blocks = tl.cdiv(seq_length, BLOCK_M)

    seq_a_block_id = num_seq_blocks - seq_alloc_prog_id - 1
    seq_b_block_id = seq_alloc_prog_id - (num_seq_alloc_progs - num_seq_blocks)

    if seq_a_block_id >= 0 or seq_b_block_id >= 0:
        # Universal stuff
        qk_scale = inv_log2 * logit_scale
        M_range = tl.arange(0, BLOCK_M)
        N_range = tl.arange(0, BLOCK_N)
        D_range = tl.arange(0, BLOCK_D)
        D_mask = D_range < head_size
        if not use_cumsum:
            cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(
                Q_ptr.type.element_ty)
        else:
            cm = None

        if seq_a_block_id >= 0:
            # First head block
            head_id = fhead_id * 2
            Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
            K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
            V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
            O_head_seq_ptr = O_ptr + stride_oh * head_id + stride_om * seq_start_offset
            R_head_seq_ptr = R_ptr + stride_rh * head_id + stride_rm * seq_start_offset
            A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
            W_head_seq_ptr = W_ptr + stride_wh * head_id + stride_am * seq_start_offset
            _forward_one_row(
                seq_a_block_id,
                seq_length,
                qk_scale,
                M_range,
                N_range,
                D_range,
                D_mask,
                cm,
                Q_head_seq_ptr,
                stride_qm,
                stride_qd,
                K_head_seq_ptr,
                stride_kn,
                stride_kd,
                V_head_seq_ptr,
                stride_vn,
                stride_vd,
                O_head_seq_ptr,
                stride_om,
                stride_od,
                R_head_seq_ptr,
                stride_rm,
                A_head_seq_ptr,
                stride_am,
                W_head_seq_ptr,
                stride_wm,
                stride_wn,
                BLOCK_D,
                NO_D_MASK,
                NO_M_MASK,
                NO_N_MASK,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                no_grad,
                acc_dtype,
                return_attention,
                use_cumsum=use_cumsum,
                attend_current=attend_current
            )
        if seq_b_block_id >= 0 and fhead_id * 2 + 1 < num_heads:
            # Reverse head block
            head_id = fhead_id * 2 + 1
            Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
            K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
            V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
            O_head_seq_ptr = O_ptr + stride_oh * head_id + stride_om * seq_start_offset
            R_head_seq_ptr = R_ptr + stride_rh * head_id + stride_rm * seq_start_offset
            A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
            W_head_seq_ptr = W_ptr + stride_wh * head_id + stride_am * seq_start_offset
            _forward_one_row(
                seq_b_block_id,
                seq_length,
                qk_scale,
                M_range,
                N_range,
                D_range,
                D_mask,
                cm,
                Q_head_seq_ptr,
                stride_qm,
                stride_qd,
                K_head_seq_ptr,
                stride_kn,
                stride_kd,
                V_head_seq_ptr,
                stride_vn,
                stride_vd,
                O_head_seq_ptr,
                stride_om,
                stride_od,
                R_head_seq_ptr,
                stride_rm,
                A_head_seq_ptr,
                stride_am,
                W_head_seq_ptr,
                stride_wm,
                stride_wn,
                BLOCK_D,
                NO_D_MASK,
                NO_M_MASK,
                NO_N_MASK,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                no_grad,
                acc_dtype,
                return_attention,
                use_cumsum=use_cumsum,
                attend_current=attend_current
            )


def varlen_fwd(
    q, k, v, cu_seqlens, max_seqlens, logit_scale, attend_current=False, no_grad=False, return_attention=False, BLOCK_M=64, BLOCK_N=32
):
    batch_size = cu_seqlens.size(0)
    num_heads, token_size, dim_size = q.size()
    o = torch.empty_like(q)
    rem = torch.zeros_like(q[:, :, 0], device=q.device)
    neg_log_acc = torch.zeros_like(rem, device=q.device, dtype=torch.float32)
    if return_attention:
        W = torch.full((num_heads, token_size, token_size), 0.0,
                       dtype=torch.float32, device=q.device)
    else:
        W = torch.empty((1, 1, 1), device=q.device)

    _compileable_forward(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlens,
        logit_scale,
        no_grad,
        return_attention,
        BLOCK_M,
        BLOCK_N,
        num_heads,
        batch_size,
        token_size,
        dim_size,
        o,
        rem,
        neg_log_acc,
        W,
        attend_current=attend_current
    )
    if return_attention:
        return o, rem, neg_log_acc, W
    else:
        return o, rem, neg_log_acc


@custom_op("varlen_fwd", mutates_args={"o", "rem", "neg_log_acc", "W"})
def _compileable_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlens: int,
    logit_scale: float,
    no_grad: bool,
    return_attention: bool,
    BLOCK_M: int,
    BLOCK_N: int,
    num_heads: int,
    batch_size: int,
    token_size: int,
    dim_size: int,
    o: torch.Tensor,
    rem: torch.Tensor,
    neg_log_acc: torch.Tensor,
    W: torch.Tensor,
    attend_current: bool,
) -> None:
    num_sequences = batch_size
    num_folded_heads = triton.cdiv(num_heads, 2)
    num_seq_blocks = triton.cdiv(max_seqlens, BLOCK_M) + 1
    BLOCK_D = triton.next_power_of_2(dim_size)
    grid = (num_sequences, num_folded_heads, num_seq_blocks)
    q_stride = q.stride()
    k_stride = k.stride()
    v_stride = v.stride()
    o_stride = o.stride()

    _forward[grid](
        q, q_stride[0], q_stride[1], q_stride[2],
        k, k_stride[0], k_stride[1], k_stride[2],
        v, v_stride[0], v_stride[1], v_stride[2],
        o, o_stride[0], o_stride[1], o_stride[2],
        rem,
        rem.stride(0),
        rem.stride(1),
        neg_log_acc,
        neg_log_acc.stride(0),
        neg_log_acc.stride(1),
        W,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        cu_seqlens,
        # pid_debug,
        logit_scale=logit_scale,
        batch_size=batch_size,
        token_size=token_size,
        head_size=dim_size,
        num_heads=num_heads,
        no_grad=no_grad,
        BLOCK_D=BLOCK_D,
        NO_D_MASK=BLOCK_D == dim_size,
        NO_M_MASK=False,
        NO_N_MASK=False,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        ALLOW_TF32=ALLOW_TF32,
        inv_log2=inv_log2,
        return_attention=return_attention,
        acc_dtype=tl.float32,
        use_cumsum=False,
        attend_current=attend_current
    )
