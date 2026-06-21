"""Agent Protocol — interface for tree search. MockAgent now, LLaVA-Med later."""
from typing import Protocol, Union, runtime_checkable
from src.contracts import Action, TreeNode, Query


@runtime_checkable
class Agent(Protocol):
    def propose_actions(self, node: TreeNode, query: Query, k: int) -> list[Union[Action, str]]:
        """Return up to k candidate Actions, or an Answer string."""
        ...
