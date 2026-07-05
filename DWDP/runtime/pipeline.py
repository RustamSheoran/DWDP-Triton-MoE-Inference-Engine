from __future__ import annotations

from dataclasses import dataclass

from DWDP.comms_planner import CommunicationPlan
from DWDP.dispatcher import DispatchPlan
from DWDP.executor import ExecutorOutput
from DWDP.merger import MergerOutput
from DWDP.router import RouterOutput
from DWDP.scheduler import ExecutionPlan

from .profiler import RuntimeProfile


@dataclass(slots=True)
class RuntimePipelineOutput:
    """Strongly typed output for one complete DWDP MoE pipeline invocation."""

    router_output: RouterOutput
    dispatch_plan: DispatchPlan
    execution_plan: ExecutionPlan
    communication_plan: CommunicationPlan
    executor_output: ExecutorOutput
    merger_output: MergerOutput
    profile: RuntimeProfile | None = None

    @property
    def hidden_states(self):
        """Final hidden states produced by the Merger."""

        return self.merger_output.hidden_states
