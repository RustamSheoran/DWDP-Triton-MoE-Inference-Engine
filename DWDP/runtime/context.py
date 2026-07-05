from __future__ import annotations

from dataclasses import dataclass

from DWDP.comms_planner import CommunicationPlannerWorkspace
from DWDP.dispatcher import DispatchWorkspace
from DWDP.executor import ExecutorWorkspace
from DWDP.merger import MergerWorkspace
from DWDP.scheduler import SchedulerWorkspace

from .config import RuntimeConfig


@dataclass(slots=True)
class RuntimeWorkspaces:
    """Reusable workspaces owned by the runtime orchestration layer."""

    dispatch: DispatchWorkspace
    scheduler: SchedulerWorkspace
    comms: CommunicationPlannerWorkspace
    executor: ExecutorWorkspace
    merger: MergerWorkspace

    @classmethod
    def create(cls) -> "RuntimeWorkspaces":
        """Create one workspace object per stateful DWDP stage."""

        return cls(
            dispatch=DispatchWorkspace(),
            scheduler=SchedulerWorkspace(),
            comms=CommunicationPlannerWorkspace(),
            executor=ExecutorWorkspace(),
            merger=MergerWorkspace(),
        )

    def estimated_bytes(self) -> int:
        """Estimate total bytes held by reusable runtime workspaces."""

        return (
            self.dispatch.estimated_bytes()
            + self.scheduler.estimated_bytes()
            + self.comms.estimated_bytes()
            + self.executor.estimated_bytes()
            + self.merger.estimated_bytes()
        )


@dataclass(slots=True)
class RuntimeContext:
    """Execution context shared across one runtime instance."""

    config: RuntimeConfig
    workspaces: RuntimeWorkspaces | None = None

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "RuntimeContext":
        """Build runtime context and optional workspaces from config."""

        return cls(
            config=config,
            workspaces=RuntimeWorkspaces.create() if config.enable_workspace else None,
        )
