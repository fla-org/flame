import math

import torch
import triton
import triton.language as tl

from ..utils import ALLOW_TF32, inv_log2
from .sb_varlen_fwd import compute_block, load_kv

from ..utils import custom_op


@triton.jit
# TODO add two-step lock?
def locked_add(Lock_ptr, Count_ptr, A_ptrs, a, B_ptrs, b, N_mask, NO_N_MASK, D_mask, NO_D_MASK: tl.constexpr):
    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(Count_ptr, eviction_policy="evict_last")
    if NO_D_MASK:
        if NO_N_MASK:
            if count == 0:
                tl.store(Count_ptr, True, eviction_policy="evict_last")
            else:
                a += tl.load(A_ptrs, eviction_policy="evict_last")
                b += tl.load(B_ptrs, eviction_policy="evict_last")
            tl.store(A_ptrs, a, eviction_policy="evict_last")
            tl.store(B_ptrs, b, eviction_policy="evict_last")

        else:
            if count == 0:
                tl.store(Count_ptr, True, eviction_policy="evict_last")
            else:
                a += tl.load(A_ptrs,
                             mask=N_mask[:, None], eviction_policy="evict_last")
                b += tl.load(B_ptrs,
                             mask=N_mask[:, None], eviction_policy="evict_last")
            tl.store(A_ptrs, a, mask=N_mask[:, None],
                     eviction_policy="evict_last")
            tl.store(B_ptrs, b, mask=N_mask[:, None],
                     eviction_policy="evict_last")

    else:
        mask = N_mask[:, None] & D_mask[None, :]
        if count == 0:
            tl.store(Count_ptr, True, eviction_policy="evict_last")
        else:
            a += tl.load(A_ptrs, mask=mask, eviction_policy="evict_last")
            b += tl.load(B_ptrs, mask=mask, eviction_policy="evict_last")
        tl.store(A_ptrs, a, mask=mask, eviction_policy="evict_last")
        tl.store(B_ptrs, b, mask=mask, eviction_policy="evict_last")

    tl.atomic_xchg(Lock_ptr, 0)


def get_configs():
    return [triton.Config({"BLOCK_M": mb, "BLOCK_N": nb}, num_stages=s, num_warps=w)
            for mb in [64, 128]
            for nb in [16, 32, 64]
            for s in [8, 7, 6, 5, 4, 3, 2]
            for w in [4, 2]]


@triton.autotune(
    configs=get_configs(),
    key=["token_size", "head_size"],
    reset_to_zero=["DK_ptr", "DV_ptr"]
)
@triton.jit
def _backward(
    DO_ptr,
    stride_doh: tl.constexpr,
    stride_dom,
    stride_dod: tl.constexpr,
    DR_ptr,
    stride_drh,
    stride_drm,
    A_ptr,
    stride_ah,
    stride_am,
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
    DQ_ptr,
    stride_dqh: tl.constexpr,
    stride_dqm,
    stride_dqd: tl.constexpr,
    DK_ptr,
    stride_dkh: tl.constexpr,
    stride_dkn,
    stride_dkd: tl.constexpr,
    DV_ptr,
    stride_dvh: tl.constexpr,
    stride_dvn,
    stride_dvd: tl.constexpr,
    KV_Lock_ptr,
    KV_Count_ptr,
    stride_kvs,
    stride_kvh,
    CSL_ptr,
    logit_scale,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_CSL: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
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
        cm = tl.where(N_range[:, None] >= N_range[None, :],
                      1.0, 0.0).to(Q_ptr.type.element_ty)

        if seq_a_block_id >= 0:
            head_id = fhead_id * 2
            DO_head_seq_ptr = DO_ptr + stride_doh * head_id + stride_dom * seq_start_offset
            DR_head_seq_ptr = DR_ptr + stride_drh * head_id + stride_drm * seq_start_offset
            A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
            Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
            K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
            V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
            DQ_head_seq_ptr = DQ_ptr + stride_dqh * head_id + stride_dqm * seq_start_offset
            DK_head_seq_ptr = DK_ptr + stride_dkh * head_id + stride_dkn * seq_start_offset
            DV_head_seq_ptr = DV_ptr + stride_dvh * head_id + stride_dvn * seq_start_offset
            KV_Lock_head_seq_ptr = KV_Lock_ptr + stride_kvs * seq_id + stride_kvh * head_id
            KV_Count_head_seq_ptr = KV_Count_ptr + \
                stride_kvs * seq_id + stride_kvh * head_id
            _backward_one_row(
                seq_a_block_id,
                seq_length,
                qk_scale,
                M_range,
                N_range,
                D_range,
                D_mask,
                cm,
                DO_head_seq_ptr,
                stride_dom,
                stride_dod,
                DR_head_seq_ptr,
                stride_drm,
                A_head_seq_ptr,
                stride_am,
                Q_head_seq_ptr,
                stride_qm,
                stride_qd,
                K_head_seq_ptr,
                stride_kn,
                stride_kd,
                V_head_seq_ptr,
                stride_vn,
                stride_vd,
                DQ_head_seq_ptr,
                stride_dqm,
                stride_dqd,
                DK_head_seq_ptr,
                stride_dkn,
                stride_dkd,
                DV_head_seq_ptr,
                stride_dvn,
                stride_dvd,
                KV_Lock_head_seq_ptr,
                KV_Count_head_seq_ptr,
                logit_scale,
                BLOCK_D,
                NO_D_MASK,
                NO_M_MASK,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                acc_dtype,
                attend_current=attend_current
            )
        if seq_b_block_id >= 0 and fhead_id * 2 + 1 < num_heads:
            head_id = fhead_id * 2 + 1
            DO_head_seq_ptr = DO_ptr + stride_doh * head_id + stride_dom * seq_start_offset
            DR_head_seq_ptr = DR_ptr + stride_drh * head_id + stride_drm * seq_start_offset
            A_head_seq_ptr = A_ptr + stride_ah * head_id + stride_am * seq_start_offset
            Q_head_seq_ptr = Q_ptr + stride_qh * head_id + stride_qm * seq_start_offset
            K_head_seq_ptr = K_ptr + stride_kh * head_id + stride_kn * seq_start_offset
            V_head_seq_ptr = V_ptr + stride_vh * head_id + stride_vn * seq_start_offset
            DQ_head_seq_ptr = DQ_ptr + stride_dqh * head_id + stride_dqm * seq_start_offset
            DK_head_seq_ptr = DK_ptr + stride_dkh * head_id + stride_dkn * seq_start_offset
            DV_head_seq_ptr = DV_ptr + stride_dvh * head_id + stride_dvn * seq_start_offset
            KV_Lock_head_seq_ptr = KV_Lock_ptr + stride_kvs * seq_id + stride_kvh * head_id
            KV_Count_head_seq_ptr = KV_Count_ptr + \
                stride_kvs * seq_id + stride_kvh * head_id
            _backward_one_row(
                seq_b_block_id,
                seq_length,
                qk_scale,
                M_range,
                N_range,
                D_range,
                D_mask,
                cm,
                DO_head_seq_ptr,
                stride_dom,
                stride_dod,
                DR_head_seq_ptr,
                stride_drm,
                A_head_seq_ptr,
                stride_am,
                Q_head_seq_ptr,
                stride_qm,
                stride_qd,
                K_head_seq_ptr,
                stride_kn,
                stride_kd,
                V_head_seq_ptr,
                stride_vn,
                stride_vd,
                DQ_head_seq_ptr,
                stride_dqm,
                stride_dqd,
                DK_head_seq_ptr,
                stride_dkn,
                stride_dkd,
                DV_head_seq_ptr,
                stride_dvn,
                stride_dvd,
                KV_Lock_head_seq_ptr,
                KV_Count_head_seq_ptr,
                logit_scale,
                BLOCK_D,
                NO_D_MASK,
                NO_M_MASK,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                acc_dtype,
                attend_current=attend_current
            )


@triton.jit
def _backward_one_row(
    seq_prog_id,
    seq_length,
    qk_scale,
    M_range,
    N_range,
    D_range,
    D_mask,
    cm,
    DO_head_seq_ptr,
    stride_dom,
    stride_dod: tl.constexpr,
    DR_head_seq_ptr,
    stride_drm,
    A_head_seq_ptr,
    stride_am: tl.constexpr,
    Q_head_seq_ptr,
    stride_qm,
    stride_qd: tl.constexpr,
    K_head_seq_ptr,
    stride_kn,
    stride_kd: tl.constexpr,
    V_head_seq_ptr,
    stride_vn,
    stride_vd: tl.constexpr,
    DQ_head_seq_ptr,
    stride_dqm,
    stride_dqd: tl.constexpr,
    DK_head_seq_ptr,
    stride_dkn,
    stride_dkd: tl.constexpr,
    DV_head_seq_ptr,
    stride_dvn,
    stride_dvd: tl.constexpr,
    KV_Lock_ptr,
    KV_Count_ptr,
    logit_scale,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
    is_compiling: tl.constexpr = False,
    attend_current: tl.constexpr = False,
):
    # Loading thread information
    block_start_offset = BLOCK_M * seq_prog_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    NO_M_MASK = (block_start_offset + BLOCK_M - 1) < seq_length

    N_blk_idxs_start = 0
    N_blk_idxs = N_blk_idxs_start + N_range

    # Init pointers
    # Inputs
    DO_blk_ptrs = DO_head_seq_ptr + \
        (stride_dom * M_blk_idxs[:, None] + stride_dod * D_range[None, :])

    K_blk_ptrs = K_head_seq_ptr + \
        (stride_kn * N_blk_idxs[:, None] + stride_kd * D_range[None, :])
    Q_blk_ptrs = Q_head_seq_ptr + \
        (stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :])
    V_blk_ptrs = V_head_seq_ptr + \
        (stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :])
    A_blk_ptrs = A_head_seq_ptr + stride_am * M_blk_idxs
    # Outputs
    DQ_blk_ptrs = DQ_head_seq_ptr + \
        (stride_dqm * M_blk_idxs[:, None] + stride_dqd * D_range[None, :])
    DK_blk_ptrs = DK_head_seq_ptr + \
        (stride_dkn * N_blk_idxs[:, None] + stride_dkd * D_range[None, :])
    DV_blk_ptrs = DV_head_seq_ptr + \
        (stride_dvn * N_blk_idxs[:, None] + stride_dvd * D_range[None, :])
    DR_blk_ptrs = DR_head_seq_ptr + stride_drm * M_blk_idxs

    # --- Load band vectors ---
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
            do = tl.load(DO_blk_ptrs)
            dr = tl.load(DR_blk_ptrs)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None])
            do = tl.load(DO_blk_ptrs, mask=M_mask[:, None])
            dr = tl.load(DR_blk_ptrs, mask=M_mask)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
    else:
        MD_mask = M_mask[:, None] & D_mask[None, :]
        q = tl.load(Q_blk_ptrs, mask=MD_mask)
        do = tl.load(DO_blk_ptrs, mask=MD_mask)
        dr = tl.load(DR_blk_ptrs, mask=M_mask)
        neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
    # --- End band vectors ---

    # Init accumulators
    neg_log_acc = neg_log_acc.to(dtype=acc_dtype)
    grad_prev_acc = tl.zeros((BLOCK_M,), dtype=acc_dtype)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=acc_dtype)

    fwd_cm = tl.trans(cm)
    # always multiple of number of blocks.
    iters = (block_start_offset + BLOCK_M) // BLOCK_N
    # if (last_N_blk_idxs_end - sequence_start_offset) % BLOCK_N > 0:
    #     tl.device_print('remainder')
    # Iterate only up to start of sequence
    for i in range(iters):
        on_band = (iters - i - 1) < BLOCK_M // BLOCK_N
        N_mask = N_blk_idxs < seq_length
        NO_N_MASK = (N_blk_idxs_start + BLOCK_N - 1) < seq_length
        # --- Recompute block ---
        k, v = load_kv(
            K_blk_ptrs,
            V_blk_ptrs,
            N_mask=N_mask,
            NO_N_MASK=(N_blk_idxs_start + BLOCK_N - 1) < seq_length,
            # N_mask=N_mask, NO_N_MASK=False,
            D_mask=D_mask,
            NO_D_MASK=NO_D_MASK,
        )
        p, log_om_beta, neg_log_acc = compute_block(
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
            backward=True,
            is_compiling=is_compiling,
        )

        if not NO_M_MASK:
            neg_log_acc = tl.where(M_mask, neg_log_acc, 0.0)

        # --- Do gradient stuff ---
        att_dA = p * \
            (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None])
        cumul_att_dA = (
            tl.dot(att_dA.to(cm.dtype), fwd_cm,
                   allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
        )  # 180 -> 174
        # cumul_att_dA = tl.cumsum(att_dA, axis=1) + grad_prev_acc[:, None] # 180 -> 174
        grad_prev_acc += tl.sum(att_dA, axis=1)
        beta = 1 - tl.exp2(log_om_beta)  # 180 -> 175
        dqk = att_dA - beta * cumul_att_dA

        dq = tl.dot(dqk.to(k.dtype), k, acc=dq, allow_tf32=ALLOW_TF32)
        block_dk = tl.dot(tl.trans(dqk).to(q.dtype), q,
                          allow_tf32=ALLOW_TF32) * logit_scale
        block_dv = tl.dot(tl.trans(p), do.to(p.dtype), allow_tf32=ALLOW_TF32)
        locked_add(
            KV_Lock_ptr + i,
            KV_Count_ptr + i,
            DK_blk_ptrs,
            block_dk,
            DV_blk_ptrs,
            block_dv,
            N_mask,
            NO_N_MASK,
            D_mask,
            NO_D_MASK,
        )
        # --- End gradient stuff ---
        N_blk_idxs += BLOCK_N
        N_blk_idxs_start += BLOCK_N
        K_blk_ptrs += BLOCK_N * stride_kn
        V_blk_ptrs += BLOCK_N * stride_vn
        DK_blk_ptrs += BLOCK_N * stride_dkn
        DV_blk_ptrs += BLOCK_N * stride_dvn

    dq = (logit_scale * dq).to(DQ_head_seq_ptr.type.element_ty)

    if NO_D_MASK:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None])
    else:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None] & D_mask[None, :])


def varlen_bwd(
    do: torch.Tensor,
    dr: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlens: int,
    neg_log_acc: torch.Tensor,
    logit_scale,
    attend_current=False,
    BLOCK_M=64,
    BLOCK_N=32,
):
    batch_size = cu_seqlens.size(0)
    num_heads, token_size, dim_size = q.size()
    if logit_scale is None:
        logit_scale = 1 / math.sqrt(dim_size)
    N_count = triton.cdiv(token_size, BLOCK_N)

    # dqdkdv = torch.zeros((token_size, num_heads, 3 * dim_size), device=do.device, dtype=do.dtype)
    # dqdkdv = dqdkdv.permute(1, 0, 2)
    # dq, dk, dv = dqdkdv.chunk(3, dim=-1)
    with torch.inference_mode():
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        num_sequences = batch_size
        num_folded_heads = triton.cdiv(num_heads, 2)
        num_seq_blocks = triton.cdiv(max_seqlens, BLOCK_M) + 1
        N_count = num_seq_blocks * (BLOCK_M // BLOCK_N)
        dkdv_lock = torch.zeros(
            (num_sequences, num_heads, N_count), dtype=torch.int32, device=q.device)
        dkdv_count = torch.zeros(
            (num_sequences, num_heads, N_count), dtype=torch.bool, device=q.device)
        _compileable_backward(
            do,
            dr,
            q,
            k,
            v,
            cu_seqlens,
            neg_log_acc,
            logit_scale,
            BLOCK_M,
            BLOCK_N,
            batch_size,
            num_heads,
            token_size,
            dim_size,
            dq,
            dk,
            dv,
            dkdv_lock,
            dkdv_count,
            num_sequences,
            num_folded_heads,
            num_seq_blocks,
            attend_current=attend_current
        )
    return dq, dk, dv


@custom_op("varlen_bwd", mutates_args={"dq", "dk", "dv", "dkdv_lock", "dkdv_count"})
def _compileable_backward(
    do: torch.Tensor,
    dr: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    neg_log_acc: torch.Tensor,
    logit_scale: float,
    BLOCK_M: int,
    BLOCK_N: int,
    batch_size: int,
    num_heads: int,
    token_size: int,
    dim_size: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dkdv_lock: torch.Tensor,
    dkdv_count: torch.Tensor,
    num_sequences: int,
    num_folded_heads: int,
    num_seq_blocks: int,
    attend_current: bool = False,
) -> None:
    BLOCK_D = triton.next_power_of_2(dim_size)
    _backward[num_sequences, num_folded_heads, num_seq_blocks](
        # DO_ptr, stride_doh, stride_dom, stride_dod,
        do,
        do.stride(0),
        do.stride(1),
        do.stride(2),
        # DR_ptr, stride_drh, stride_drm,
        dr,
        dr.stride(0),
        dr.stride(1),
        # A_ptr, stride_ah, stride_am,
        neg_log_acc,
        neg_log_acc.stride(0),
        neg_log_acc.stride(1),
        # Q_ptr, stride_qh, stride_qm, stride_qd,
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # K_ptr, stride_kh, stride_kn, stride_kd,
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V_ptr, stride_vh, stride_vn, stride_vd,
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # DQ_ptr, stride_dqh, stride_dqm, stride_dqd,
        dq,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        # DK_ptr, stride_dkh, stride_dkn, stride_dkd,
        dk,
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        # DV_ptr, stride_dvh, stride_dvn, stride_dvd,
        dv,
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        # KV_Lock_ptr, KV_Count_ptr, stride_kvl,
        dkdv_lock,
        dkdv_count,
        dkdv_lock.stride(0),
        dkdv_lock.stride(1),
        cu_seqlens,
        logit_scale=logit_scale,
        batch_size=batch_size,
        token_size=token_size,
        head_size=dim_size,
        num_heads=num_heads,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_CSL=triton.next_power_of_2(batch_size),
        NO_D_MASK=BLOCK_D == dim_size,
        NO_M_MASK=False,
        NO_N_MASK=False,
        ALLOW_TF32=ALLOW_TF32,
        inv_log2=inv_log2,
        attend_current=attend_current
    )
