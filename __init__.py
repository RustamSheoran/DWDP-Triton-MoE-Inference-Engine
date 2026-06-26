"""DeepSeek-V4 architecture package.

Importing this module registers ``DeepseekV4Config`` / ``DeepseekV4ForCausalLM``
with the HuggingFace ``transformers`` auto classes so that

    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained("path/to/our/checkpoint")

resolves to ``DeepseekV4ForCausalLM`` without ``trust_remote_code=True``.
"""
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_deepseek_v4 import DeepseekV4Config
from .modeling_deepseek_v4 import (
    DeepseekV4Model,
    DeepseekV4ForCausalLM,
    DeepseekV4PreTrainedModel,
)

# Idempotent registration — safe to import twice.
try:
    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
except (ValueError, TypeError):
    pass
try:
    AutoModel.register(DeepseekV4Config, DeepseekV4Model, exist_ok=True)
except (ValueError, TypeError):
    pass
try:
    AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)
except (ValueError, TypeError):
    pass

__all__ = [
    "DeepseekV4Config",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
    "DeepseekV4PreTrainedModel",
]
