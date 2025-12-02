from typing import Dict, List, Sequence, Protocol, Any, ClassVar
from powergrid.utils.typing import Array

class AgentLike(Protocol):
    @property
    def agent_id(self) -> str: ...
    @property
    def level(self) -> int: ...

class FeatureProvider(Protocol):
    """Feature providers used in agent states."""
    visibility: ClassVar[Sequence[str]]
    def as_vector(self) -> Array: ...
    def names(self) -> List[str]: ...
    def to_dict(self) -> Dict: ...
    def set_values(self, values: Any) -> None: ...
    @classmethod
    def from_dict(cls, d: Dict): ...
    def is_observable_by(self, agent: AgentLike, owner: AgentLike) -> Array:
        if "public" in self.visibility:
            return True
        if "owner" in self.visibility:
            return agent.agent_id == owner.agent_id
        if "system" in self.visibility:
            return agent.level >= 3
        if "upper_level" in self.visibility:
            return agent.level == owner.level + 1
        # default: treat as private
        return False
