from pydantic import BaseModel, Field
from typing import Tuple, Literal, Optional, Dict, List, Any

QType = Literal["existential", "negation", "relational", "counting", "open"]

Tier = Literal["A", "B", "ABSTAIN"]

ToolName = Literal[
    "is_a", "disjoint", "anatomy_of", "compose_laterality",
    "get_exclusion_list", "retrieve",
    "neighbors", "find_path",
    "inspect", "re_detect", "compare",
]

class PerceptualFact(BaseModel):
    """Pydantic model representing a grounded perceptual fact extracted from a chest X-ray."""
    concept: str = Field(
        ...,
        description="Canonical clinical finding name (e.g., 'cardiomegaly', 'pleural effusion')"
    )
    bbox: Tuple[float, float, float, float] = Field(
        ...,
        description="Region coordinates on the image as (x1, y1, x2, y2)"
    )
    laterality: Literal["left", "right", "bilateral", "midline"] = Field(
        ...,
        description="Laterality/location attribute of the finding"
    )
    conf: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the finding in the range [0.0, 1.0]"
    )

    class Config:
        frozen = True


class Query(BaseModel):
    """Parsed VQA question — closed schema, no answer slot."""
    type: QType
    target: Optional[str] = Field(
        None,
        description="Canonical finding name (DAG node), or None for open queries",
    )
    constraints: Dict[str, str] = Field(
        default_factory=dict,
        description='e.g. {"attr": "laterality"}, {} if none',
    )
    raw_question: str = Field(
        ..., description="Original question text, preserved for debugging/rendering"
    )
    parse_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="1.0 for rule hits, lower for LLM fallback"
    )
    parser_tier: Literal["rule", "llm"] = Field(
        ..., description="Which tier produced this parse"
    )

    class Config:
        frozen = True


class Action(BaseModel):
    """One step the agent takes — symbolic (graph op) or visual (image op)."""
    tool: ToolName
    args: Dict[str, Any] = Field(default_factory=dict)
    kind: Literal["symbolic", "visual"] = "symbolic"

    class Config:
        frozen = True


class Observation(BaseModel):
    """Result returned by a tool after executing an Action."""
    result: Any = None
    ok: bool = True

    class Config:
        frozen = True


class TreeNode(BaseModel):
    """One node in the search tree — a reasoning state."""
    state_facts: List[PerceptualFact] = Field(default_factory=list)
    history: List[Tuple[Action, Observation]] = Field(default_factory=list)
    answer: Optional[str] = None
    reward: float = 0.0
    parent_id: Optional[int] = None
    reflection: str = ""


class SearchResult(BaseModel):
    """Output of tree search — answer + tier + the winning path (trace)."""
    answer: str = ""
    tier: Tier = "ABSTAIN"
    path: List[Tuple[Action, Observation]] = Field(default_factory=list)
    conf: float = 0.0

    class Config:
        frozen = True
