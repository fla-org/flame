import torch
import triton
import triton.language as tl

from ..utils import ALLOW_TF32, inv_log2, custom_op
from ..sb_varlen.sb_varlen_fwd import _forward_one_row
from ..sb_varlen.softplus import softplus



def get_configs():
    return [triton.Config({}, num_stages=s, num_warps=w) for s in [4] for w in [4]]


@triton.autotune(configs=get_configs(), key=["token_size", "head_size"])
@triton.jit
def _forward(
    Q_ptr,
    stride_qb,
    stride_qh,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    K_ptr,
    stride_kb,
    stride_kh,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    V_ptr,
    stride_vb,
    stride_vh,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    O_ptr,
    stride_ob,
    stride_oh,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    R_ptr,
    stride_rb,
    stride_rh,
    stride_rm: tl.constexpr,
    A_ptr,
    stride_ab,
    stride_ah,
    stride_am: tl.constexpr,
    W_ptr,
    stride_wb,
    stride_wh,
    stride_wm,
    stride_wn,
    logit_scale: tl.constexpr,
    attend_current: tl.constexpr,
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
    is_compiling: tl.constexpr = False,
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    batch_id = tl.program_id(0)
    head_pid = tl.program_id(1)
    prog_id = tl.program_id(2)
    tl.num_programs(2)
    seq_length = token_size
    # Universal stuff
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :],
                  1.0, 0.0).to(Q_ptr.type.element_ty)

    # First head block
    head_id = head_pid
    seq_prog_id = prog_id
    # tl.store(pid_debug_ptr + head_id * tl.num_programs(1) + prog_id_start_offset + seq_prog_id, pid)
    Q_head_seq_ptr = Q_ptr + stride_qb * batch_id + stride_qh * head_id
    K_head_seq_ptr = K_ptr + stride_kb * batch_id + stride_kh * head_id
    V_head_seq_ptr = V_ptr + stride_vb * batch_id + stride_vh * head_id
    O_head_seq_ptr = O_ptr + stride_ob * batch_id + stride_oh * head_id
    R_head_seq_ptr = R_ptr + stride_rb * batch_id + stride_rh * head_id
    A_head_seq_ptr = A_ptr + stride_ab * batch_id + stride_ah * head_id
    W_head_seq_ptr = W_ptr + stride_wb * batch_id + stride_wh * head_id
    _forward_one_row(
        seq_prog_id,
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
        attend_current=attend_current,
        is_compiling=is_compiling,
    )


def _fwd(q, k, v, logit_scale,
         attend_current=False,
         no_grad=False, return_attention=False,
         BLOCK_M: int = 64, BLOCK_N: int = 32):
    batch_size, num_heads, token_size, dim_size = q.size()
    o = torch.empty_like(q)
    rem = torch.zeros_like(q[:, :, :, 0], device=q.device)
    neg_log_acc = torch.zeros_like(rem, device=q.device, dtype=torch.float32)
    if return_attention:
        W = torch.full((batch_size, num_heads, token_size, token_size),
                       0.0, dtype=torch.float32, device=q.device)
    else:
        W = torch.empty((1, 1, 1, 1), device=q.device)
    _compileable_fwd(
        q,
        k,
        v,
        logit_scale,
        no_grad,
        return_attention,
        BLOCK_M,
        BLOCK_N,
        batch_size,
        num_heads,
        token_size,
        dim_size,
        o,
        rem,
        neg_log_acc,
        W,
        attend_current=attend_current,
    )
    if return_attention:
        return o, rem, neg_log_acc, W
    else:
        return o, rem, neg_log_acc


@custom_op("attn_fwd", mutates_args={"o", "rem", "neg_log_acc", "W"})
def _compileable_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: float,
    no_grad: bool,
    return_attention: bool,
    BLOCK_M: int,
    BLOCK_N: int,
    batch_size: int,
    num_heads: int,
    token_size: int,
    dim_size: int,
    o: torch.Tensor,
    rem: torch.Tensor,
    neg_log_acc: torch.Tensor,
    W: torch.Tensor,
    attend_current: bool,
) -> None:
    num_folded_heads = num_heads
    num_seq_blocks = triton.cdiv(token_size, BLOCK_M)
    BLOCK_D = triton.next_power_of_2(dim_size)
    grid = (batch_size, num_folded_heads, num_seq_blocks)
    _forward[grid](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        rem,
        rem.stride(0),
        rem.stride(1),
        rem.stride(2),
        neg_log_acc,
        neg_log_acc.stride(0),
        neg_log_acc.stride(1),
        neg_log_acc.stride(2),
        W,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        W.stride(3),
        logit_scale=logit_scale,
        batch_size=batch_size,
        token_size=token_size,
        head_size=dim_size,
        num_heads=num_heads,
        no_grad=no_grad,
        attend_current=attend_current,
        BLOCK_D=BLOCK_D,
        NO_D_MASK=BLOCK_D == dim_size,
        NO_M_MASK=(token_size % BLOCK_M) == 0,
        NO_N_MASK=(token_size % BLOCK_N) == 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        ALLOW_TF32=ALLOW_TF32,
        inv_log2=inv_log2,
        return_attention=return_attention,
        acc_dtype=tl.float32,
        is_compiling=False,
    )
