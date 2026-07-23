from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from DWDP.dispatcher import DispatchMetadata, DispatchPlan, ExpertAssignments  # noqa: E402
from DWDP.executor.kernels import grouped_matmul, grouped_matmul_from_dispatch, reference_grouped_matmul  # noqa: E402
from DWDP.executor.weights import ExpertMajorMatrixView  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped Triton matmul requires CUDA")


def make_offsets(counts: tuple[int, ...], device: str) -> torch.Tensor:
    counts_tensor = torch.tensor(counts, dtype=torch.int64, device=device)
    return torch.cat((counts_tensor.new_zeros(1), counts_tensor.cumsum(dim=0)), dim=0)


def make_dispatch_plan(offsets: torch.Tensor) -> DispatchPlan:
    num_assignments = int(offsets[-1].item())
    num_experts = offsets.numel() - 1
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int64, device=offsets.device),
        offsets[1:] - offsets[:-1],
    )
    assignments = ExpertAssignments(
        expert_ids=expert_ids,
        packed_token_indices=torch.arange(num_assignments, dtype=torch.int64, device=offsets.device),
        packed_routing_weights=torch.ones(num_assignments, dtype=torch.float16, device=offsets.device),
    )
    return DispatchPlan(
        assignments=assignments,
        metadata=DispatchMetadata(
            num_tokens=num_assignments,
            num_assignments=num_assignments,
            num_experts=num_experts,
            top_k=1,
            token_shape=(num_assignments,),
            expert_counts=offsets[1:] - offsets[:-1],
            expert_offsets=offsets,
            token_permutation=torch.arange(num_assignments, dtype=torch.int64, device=offsets.device),
            inverse_permutation=torch.arange(num_assignments, dtype=torch.int64, device=offsets.device),
            destination_positions=torch.arange(num_assignments, dtype=torch.int64, device=offsets.device),
            stable_order=True,
            algorithm="test",
        ),
    )


@pytest.mark.parametrize("counts", [(3, 5), (0, 7, 2, 11), (33, 1, 17, 0)])
@pytest.mark.parametrize("hidden_size,output_size", [(32, 48), (64, 96)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_matmul_matches_pytorch_reference(
    counts: tuple[int, ...],
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("BF16 Tensor Core validation requires Ampere or newer")
    torch.manual_seed(1234)
    offsets = make_offsets(counts, "cuda")
    assignments = int(offsets[-1].item())
    expert_inputs = torch.randn(assignments, hidden_size, device="cuda", dtype=dtype)
    expert_weights = torch.randn(len(counts), output_size, hidden_size, device="cuda", dtype=dtype)

    reference = reference_grouped_matmul(expert_inputs, expert_weights, offsets)
    actual = grouped_matmul(expert_inputs, expert_weights, offsets)

    tolerance = 3e-2 if dtype == torch.bfloat16 else 2e-2
    assert torch.allclose(actual, reference, rtol=tolerance, atol=tolerance)
    assert (actual - reference).abs().max().item() < (0.5 if dtype == torch.bfloat16 else 0.1)


def test_grouped_matmul_preserves_deterministic_expert_major_order() -> None:
    offsets = make_offsets((2, 3, 1), "cuda")
    expert_inputs = torch.randn(6, 32, device="cuda", dtype=torch.float16)
    expert_weights = torch.randn(3, 16, 32, device="cuda", dtype=torch.float16)

    first = grouped_matmul(expert_inputs, expert_weights, offsets)
    second = grouped_matmul(expert_inputs, expert_weights, offsets)

    assert torch.equal(first, second)


def test_grouped_matmul_consumes_dispatch_offsets_and_logical_weight_view() -> None:
    offsets = make_offsets((2, 1), "cuda")
    expert_inputs = torch.randn(3, 32, device="cuda", dtype=torch.float16)
    physical_weights = torch.randn(2, 24, 32, device="cuda", dtype=torch.float16)
    view = ExpertMajorMatrixView(
        expert_ids=(0, 1),
        expert_weights=(physical_weights[0], physical_weights[1]),
        name="test_projection",
    )

    actual = grouped_matmul_from_dispatch(expert_inputs, view, make_dispatch_plan(offsets))
    reference = reference_grouped_matmul(expert_inputs, physical_weights, offsets)

    assert torch.allclose(actual, reference, rtol=2e-2, atol=2e-2)
