from .config_sba import SBAConfig
from .modeling_sba import SBAForCausalLM, SBAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = ['SBAConfig', 'SBAForCausalLM', 'SBAModel']

AutoConfig.register('sba', SBAConfig)
AutoModel.register(SBAConfig, SBAModel)
AutoModelForCausalLM.register(SBAConfig, SBAForCausalLM)
