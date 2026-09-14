"""
core/schemas.py
Canonical entity definitions for ApplyX.

All cross-module application, opportunity, matching, and trace payloads should
use these structures or their JSON-safe dictionary forms.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RequirementType(str, Enum):
    HARD = "HARD"
    PREFERRED = "PREFERRED"
    CONDITIONAL = "CONDITIONAL"


class RequirementCategory(str, Enum):
    EDUCATION = "EDUCATION"
    CGPA = "CGPA"
    GRADUATION_YEAR = "GRADUATION_YEAR"
    ENROLLMENT_STATUS = "ENROLLMENT_STATUS"
    BACKGROUND_CHECK = "BACKGROUND_CHECK"
    SKILL = "SKILL"
    WORK_AUTH = "WORK_AUTH"
    EXPERIENCE = "EXPERIENCE"
    GENERAL = "GENERAL"


class SkillRelation(str, Enum):
    DIRECT = "DIRECT"
    RELATED = "RELATED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class Requirement:
    raw_text: str
    category: RequirementCategory
    req_type: RequirementType
    operator: Optional[str] = None
    target_value: Optional[Any] = None
    secondary_category: Optional[RequirementCategory] = None
    secondary_target: Optional[Any] = None
    secondary_operator: Optional[str] = None


@dataclass
class Evidence:
    source: str
    field: str
    value: Any
    confidence: float = 1.0


@dataclass
class SubPathEvaluation:
    path_name: str
    status: str  # PASS / FAIL / UNKNOWN
    reason: str


@dataclass
class RequirementEvaluation:
    requirement: Requirement
    status: str  # PASS / FAIL / UNCERTAIN
    relation: SkillRelation
    blocking: bool
    evidence: List[Evidence] = field(default_factory=list)
    sub_paths: List[SubPathEvaluation] = field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0


@dataclass
class CanonicalOpportunity:
    id: str
    title: str
    organization: str
    type: str
    location: Optional[str] = None
    deadline: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    hard_requirements: List[str] = field(default_factory=list)
    preferred_requirements: List[str] = field(default_factory=list)
    conditional_requirements: List[str] = field(default_factory=list)
    source: str = "Direct"
    source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class MatchAnalysisResult:
    formal_eligibility: str  # YES / NO / UNCERTAIN
    hard_satisfied: int
    hard_total: int
    pref_satisfied: int
    pref_total: int
    conditional_total: int
    fit_percentage: int
    evidence_coverage: float
    gap_severity: str  # NONE / LOW / MODERATE / CRITICAL
    risk_level: str  # LOW / MEDIUM / HIGH / CRITICAL
    reasoning_confidence: float = 0.85
    reasoning_confidence_label: str = "HIGH"
    missing_hard: List[str] = field(default_factory=list)
    missing_pref: List[str] = field(default_factory=list)
    uncertain_criteria: List[str] = field(default_factory=list)
    conditional_criteria: List[str] = field(default_factory=list)
    skill_evaluations: Dict[str, Tuple[SkillRelation, str]] = field(default_factory=dict)
    evaluations: List[RequirementEvaluation] = field(default_factory=list)
    breakdown_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(asdict(self))


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/enums/containers to JSON-safe values."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
