import torch
import triton
from triton import language as tl


def _generate_asm(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 15.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p ex2.approx.ftz.f32 ${out_reg}, ${in_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_constraints(num_pack):
    return ",".join("=r" for i in range(num_pack)) + "," + ",".join("r" for i in range(num_pack))


NUM_REG: tl.constexpr = 1
asm_str: tl.constexpr = _generate_asm(NUM_REG)
constraints_str: tl.constexpr = _generate_constraints(NUM_REG)


@triton.jit
def softplus(x, is_compiling: tl.constexpr = False):
    if is_compiling:
        tl.static_print("Using triton softplus.")
        out = tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)
        return out
    else:
        out = tl.inline_asm_elementwise(
            asm=asm_str,
            constraints=constraints_str,
            pack=NUM_REG,
            args=[
                x,
            ],
            dtype=tl.float32,
            is_pure=True,
        )
        return out
