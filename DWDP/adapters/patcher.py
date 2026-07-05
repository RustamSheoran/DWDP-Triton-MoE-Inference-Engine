from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(slots=True)
class PatchRecord:
    """Reversible module replacement record."""

    name: str
    parent: nn.Module
    child_name: str
    original: nn.Module
    replacement: nn.Module


class ModulePatcher:
    """Applies and reverses Hugging Face module replacements."""

    def __init__(self) -> None:
        self.records: list[PatchRecord] = []

    def replace(self, *, name: str, parent: nn.Module, child_name: str, replacement: nn.Module) -> None:
        """Replace `parent.child_name` with `replacement` and store original."""

        original = getattr(parent, child_name)
        setattr(parent, child_name, replacement)
        self.records.append(
            PatchRecord(
                name=name,
                parent=parent,
                child_name=child_name,
                original=original,
                replacement=replacement,
            )
        )

    def restore(self) -> int:
        """Restore replacements in reverse order."""

        count = 0
        while self.records:
            record = self.records.pop()
            setattr(record.parent, record.child_name, record.original)
            count += 1
        return count
