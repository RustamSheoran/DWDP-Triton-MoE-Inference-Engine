from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from DWDP.adapters import HuggingFaceAdapter, Qwen15MoEAdapter, detect_adapter_class  # noqa: E402
from DWDP.adapters.extractor import discover_qwen_moe_layers  # noqa: E402
from DWDP.adapters.validator import compare_outputs, generated_token_parity  # noqa: E402
from DWDP.runtime import RuntimeConfig  # noqa: E402


class FakeConfig:
    model_type = "qwen2_moe"
    architectures = ("Qwen2MoeForCausalLM",)
    num_experts_per_tok = 1


class FakeQwenMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)]
        )
        self.top_k = 1
        self.norm_topk_prob = True
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
            self.experts[0].weight.copy_(torch.eye(4) * 2.0)
            self.experts[1].weight.copy_(torch.eye(4) * 3.0)

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states, self.gate(hidden_states)


class FakeQwenModel(nn.Module):
    def __init__(self, num_layers: int = 1) -> None:
        super().__init__()
        self.config = FakeConfig()
        self.layers = nn.ModuleList([nn.Module() for _ in range(num_layers)])
        for layer in self.layers:
            layer.mlp = FakeQwenMoeBlock()

    def forward(self, hidden_states: torch.Tensor):
        return self.layers[0].mlp(hidden_states)

    def generate(self, *args, **kwargs):
        del args, kwargs
        return torch.tensor([[1, 2, 3]])


def test_detects_qwen_adapter_from_model_config() -> None:
    model = FakeQwenModel()

    assert detect_adapter_class(model) is Qwen15MoEAdapter


def test_discovers_qwen_moe_layers() -> None:
    specs = discover_qwen_moe_layers(FakeQwenModel())

    assert len(specs) == 1
    assert specs[0].name == "layers.0.mlp"
    assert specs[0].num_experts == 2
    assert specs[0].top_k == 1


def test_qwen_adapter_patches_and_restores_model() -> None:
    model = FakeQwenModel()
    original = model.layers[0].mlp
    adapter = Qwen15MoEAdapter(model=model, config=RuntimeConfig())

    count = adapter.patch_model()

    assert count == 1
    assert model.layers[0].mlp is not original
    assert adapter.restore_model() == 1
    assert model.layers[0].mlp is original


def test_qwen_adapter_shares_gate_parameter_storage() -> None:
    model = FakeQwenModel()
    original_gate = model.layers[0].mlp.gate
    adapter = Qwen15MoEAdapter(model=model, config=RuntimeConfig())

    adapter.patch_model()
    patched = model.layers[0].mlp

    assert patched.router.weight.data_ptr() == original_gate.weight.data_ptr()


def test_qwen_adapter_shares_one_workspace_across_moe_layers() -> None:
    model = FakeQwenModel(num_layers=3)
    adapter = Qwen15MoEAdapter(model=model, config=RuntimeConfig())

    adapter.patch_model()

    contexts = [layer.mlp.context for layer in model.layers]
    assert contexts[0] is contexts[1] is contexts[2]


def test_shared_workspace_does_not_alias_sequential_layer_outputs() -> None:
    model = FakeQwenModel(num_layers=2)
    adapter = Qwen15MoEAdapter(model=model, config=RuntimeConfig())
    adapter.patch_model()
    hidden_states = torch.tensor(
        [[[2.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0]]]
    )

    with torch.inference_mode():
        output = hidden_states
        for layer in model.layers:
            output, _ = layer.mlp(output)

    expected = torch.tensor(
        [[[8.0, 4.0, 0.0, 0.0], [9.0, 36.0, 0.0, 0.0]]]
    )
    assert torch.equal(output, expected)


def test_patched_pytorch_block_avoids_assignment_major_output_buffers() -> None:
    model = FakeQwenModel()
    adapter = Qwen15MoEAdapter(model=model, config=RuntimeConfig())
    adapter.patch_model()
    patched = model.layers[0].mlp

    with torch.inference_mode():
        output, _ = patched(torch.randn(1, 8, 4))

    assert output.shape == (1, 8, 4)
    assert patched.context.workspaces.executor.packed_expert_outputs is None
    assert patched.context.workspaces.executor.weighted_expert_outputs is None
    assert not hasattr(patched.context.workspaces.executor, "merged_expert_outputs")
    assert patched.context.workspaces.merger.estimated_bytes() == 0


def test_patched_qwen_block_executes_dwdp_pipeline() -> None:
    model = FakeQwenModel()
    adapter = Qwen15MoEAdapter(model=model, config=RuntimeConfig())
    adapter.patch_model()
    hidden_states = torch.tensor([[[2.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0]]])

    output, router_logits = model(hidden_states)

    expected = torch.tensor([[[4.0, 2.0, 0.0, 0.0], [3.0, 12.0, 0.0, 0.0]]])
    assert torch.allclose(output, expected)
    assert router_logits.shape == (1, 2, 2)


def test_shared_expert_is_combined_after_routed_output() -> None:
    model = FakeQwenModel()
    block = model.layers[0].mlp
    block.shared_expert = nn.Linear(4, 4, bias=False)
    block.shared_expert_gate = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        block.shared_expert.weight.copy_(torch.eye(4))
        block.shared_expert_gate.weight.zero_()
    adapter = Qwen15MoEAdapter(model=model, config=RuntimeConfig())
    adapter.patch_model()
    hidden_states = torch.tensor(
        [[[2.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0]]]
    )

    with torch.inference_mode():
        output, _ = model(hidden_states)

    expected = torch.tensor(
        [[[5.0, 2.5, 0.0, 0.0], [3.5, 14.0, 0.0, 0.0]]]
    )
    assert torch.equal(output, expected)


def test_huggingface_adapter_auto_patches_supported_model() -> None:
    model = FakeQwenModel()
    runtime = HuggingFaceAdapter(model=model, config=RuntimeConfig()).create_runtime()

    assert isinstance(runtime.adapter, Qwen15MoEAdapter)
    assert model.layers[0].mlp.__class__.__name__ == "DWDPMoEBlock"


def test_validator_reports_relative_error_and_token_parity() -> None:
    comparison = compare_outputs(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.01]), atol=0.1
    )

    assert comparison.allclose
    assert comparison.max_relative_error > 0.0
    assert generated_token_parity(torch.tensor([[1, 2]]), torch.tensor([[1, 2]]))


def test_unsupported_huggingface_model_delegates() -> None:
    model = nn.Identity()
    model.config = type("Config", (), {"model_type": "dense"})()
    runtime = HuggingFaceAdapter(model=model, config=RuntimeConfig()).create_runtime()

    assert runtime.forward(torch.tensor([1.0])).item() == 1.0
