# -*- coding: utf-8 -*-

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               PrepareModuleOutput,
                                               RowwiseParallel,
                                               SequenceParallel)

try:
    from torchao.float8.float8_tensor_parallel import (
        Float8ColwiseParallel, Float8RowwiseParallel, PrepareFloat8ModuleInput)
except ImportError:
    Float8ColwiseParallel = None
    Float8RowwiseParallel = None
    PrepareFloat8ModuleInput = None
from fla.modules.fused_linear_cross_entropy import LinearLossParallel
from fla.modules.mlp import SwiGLULinearParallel
from fla.modules.parallel import PrepareModuleWeight

TP_PLANS = dict()


def dispatch_tp_plan(model, loss_parallel=False, enable_float8=False):
    model_name = model.__class__.__name__
    if model_name in TP_PLANS:
        plan_obj = TP_PLANS[model_name]
    else:
        plan_obj = FlameTPPlan

    if hasattr(model, "layer_plan"):
        plan_obj.layer_plan = model.layer_plan
    if hasattr(model, "others_plan"):
        plan_obj.others_plan = model.others_plan
    return plan_obj(model, loss_parallel, enable_float8)


def register_tp_plan(name, tp_plan):
    TP_PLANS[name] = tp_plan


class FlameTPPlan:
    def __init__(
        self,
        model=None,
        loss_parallel=False,
        enable_float8=False,
    ):
        self.model = model
        self.loss_parallel = loss_parallel
        self.enable_float8 = enable_float8

        if self.enable_float8 and Float8ColwiseParallel is not None:
            (
                self.rowwise_parallel,
                self.colwise_parallel,
                self.prepare_module_input,
                self.prepare_module_output,
            ) = (
                Float8RowwiseParallel,
                Float8ColwiseParallel,
                PrepareFloat8ModuleInput,
                PrepareModuleOutput,
            )
        else:
            (
                self.rowwise_parallel,
                self.colwise_parallel,
                self.prepare_module_input,
                self.prepare_module_output,
            ) = (
                RowwiseParallel,
                ColwiseParallel,
                PrepareModuleInput,
                PrepareModuleOutput,
            )

    """
    We set two properties others_plan/layer_plan
    So that it works better for hybrid transformer models
    """

    @property
    def others_plan(self):
        return {}

    @property
    def layer_plan(self):
        return {}


class LlamaPlan(FlameTPPlan):
    def __init__(self, model=None, loss_parallel=False, enable_float8=False):
        super().__init__(model, loss_parallel, enable_float8)

    @property
    def others_plan(self):
        return {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if self.loss_parallel else Replicate(),
                use_local_output=not self.loss_parallel,
            ),
        }

    @property
    def layer_plan(self):
        plans = {
            "input_layernorm": SequenceParallel(),
            "self_attn": self.prepare_module_input(
                input_kwarg_layouts={
                    "hidden_states": Shard(1),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                },
            ),
            "self_attn.q_proj": self.colwise_parallel(),
            "self_attn.k_proj": self.colwise_parallel(),
            "self_attn.v_proj": self.colwise_parallel(),
            "self_attn.o_proj": self.rowwise_parallel(output_layouts=Shard(1)),
            "post_attention_layernorm": SequenceParallel(),
            "mlp": self.prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": self.colwise_parallel(),
            "mlp.up_proj": self.colwise_parallel(),
            "mlp.down_proj": self.rowwise_parallel(output_layouts=Shard(1)),
        }

        return plans


class FLATransformerPlan(FlameTPPlan):
    def __init__(self, model=None, loss_parallel=False, enable_float8=False):
        super().__init__(model, loss_parallel, enable_float8)
        self.base_model_prefix = getattr(model, "base_model_prefix", "model")

    @property
    def others_plan(self):
        plans = {
            f"{self.base_model_prefix}.embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            f"{self.base_model_prefix}.norm": SequenceParallel(),
        }
        if self.loss_parallel:
            plans.update({
                "lm_head": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1) if self.loss_parallel else Replicate(),
                    use_local_output=not self.loss_parallel,
                ),
            })
        else:
            plans.update({
                "lm_head": PrepareModuleWeight(layouts=Replicate()),
                "criterion": LinearLossParallel()
            })
        return plans

    @property
    def layer_plan(self):
        plans = {
            "attn_norm": SequenceParallel(),
            "attn": self.prepare_module_input(
                input_kwarg_layouts={
                    "hidden_states": Shard(1),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                },
            ),
            "attn.q_proj": self.colwise_parallel(),
            "attn.k_proj": self.colwise_parallel(),
            "attn.v_proj": self.colwise_parallel(),
            "attn.o_proj": self.rowwise_parallel(output_layouts=Shard(1)),
            "mlp_norm": SequenceParallel(),
            "mlp": self.prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": self.colwise_parallel(),
            "mlp.up_proj": self.colwise_parallel(),
            "mlp.down_proj": self.rowwise_parallel(output_layouts=Shard(1)),
            "mlp.swiglu_linear": SwiGLULinearParallel(
                output_layouts=Shard(1)
            )
        }

        return plans


register_tp_plan("LlamaForCausalLM", LlamaPlan)
register_tp_plan("LlamaDecoderLayer", LlamaPlan)

register_tp_plan("TransformerBlock", FLATransformerPlan)
register_tp_plan("TransformerForCausalLM", FLATransformerPlan)
