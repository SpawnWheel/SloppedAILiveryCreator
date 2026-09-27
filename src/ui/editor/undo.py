"""Minimal memory-aware undo / redo stack for the paint editor."""

from typing import Callable, List, Optional


class UndoCommand:
    def __init__(self, label: str, undo: Callable[[], None], redo: Callable[[], None], cost_bytes: int = 0):
        self.label = label
        self.undo = undo
        self.redo = redo
        self.cost = cost_bytes


class UndoStack:
    def __init__(self, max_items: int = 60, max_bytes: int = 1_200_000_000):
        self._done: List[UndoCommand] = []
        self._undone: List[UndoCommand] = []
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.on_change: Optional[Callable[[], None]] = None

    def push(self, cmd: UndoCommand, already_applied: bool = True) -> None:
        if not already_applied:
            cmd.redo()
        self._done.append(cmd)
        self._undone.clear()
        while len(self._done) > self.max_items or (
            sum(c.cost for c in self._done) > self.max_bytes and len(self._done) > 1
        ):
            self._done.pop(0)
        self._changed()

    def undo(self) -> Optional[str]:
        if not self._done:
            return None
        cmd = self._done.pop()
        cmd.undo()
        self._undone.append(cmd)
        self._changed()
        return cmd.label

    def redo(self) -> Optional[str]:
        if not self._undone:
            return None
        cmd = self._undone.pop()
        cmd.redo()
        self._done.append(cmd)
        self._changed()
        return cmd.label

    def can_undo(self) -> bool:
        return bool(self._done)

    def can_redo(self) -> bool:
        return bool(self._undone)

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
        self._changed()

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()
