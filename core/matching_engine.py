"""
core/matching_engine.py
Deterministic multi-dimensional requirement parser and evaluator.

Rules:
- Missing candidate evidence => UNCERTAIN, not FAIL.
- Blocking hard FAIL => formal eligibility NO.
- Blocking hard UNCERTAIN => formal eligibility UNCERTAIN.
- Preferred gaps never disqualify.
- Conditional requirements are explicit REVIEW triggers.
- Compound OR requirements use: PASS+anything=PASS; FAIL+FAIL=FAIL;
  FAIL+UNCERTAIN=UNCERTAIN; UNCERTAIN+UNCERTAIN=UNCERTAIN.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.schemas import (
    CanonicalOpportunity,
    Evidence,
    MatchAnalysisResult,
    Requirement,
    RequirementCategory,
    RequirementEvaluation,
    RequirementType,
    SkillRelation,
    SubPathEvaluation,
)


EDUCATION_ALIASES: Dict[str, List[str]] = {
    "phd": [
        r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctoral\b",
    ],
    "master": [
        r"\bm\.?tech\b", r"\bmtech\b", r"\bm\.?s\b",
        r"\bmaster(?:'s|s)?\b", r"\bm\.?e\.?\b",
    ],
    "bachelor": [
        r"\bb\.?tech\b", r"\bbtech\b", r"\bb\.?e\.?\b",
        r"\bbe\b", r"\bbachelor(?:'s|s)?\b", r"\bbs\b", r"\bb\.?sc\.?\b",
        r"\bundergraduate\b", r"\bengineering\b",
        r"\baiml\b", r"\bai\s*/\s*ml\b",
        r"\bai\s*&\s*ml\b", r"\bai\s*and\s*ml\b",
        r"\bartificial intelligence and machine learning\b",
        r"\bcomputer science\b", r"\bdata science\b",
    ],
}

# Degree/branch aliases commonly entered in compact form in candidate profiles.
PROFILE_DEGREE_DOMAIN_ALIASES: Dict[str, List[str]] = {
    "artificial intelligence": [
        "artificial intelligence", "aiml", "ai/ml", "ai & ml", "ai and ml", "artificial intelligence and machine learning",
    ],
    "machine learning": [
        "machine learning", "aiml", "ai/ml", "ai & ml", "ai and ml",
    ],
    "computer science": [
        "computer science", "computer engineering", "cse", "cs",
    ],
}

CONDITIONAL_PHRASES = (
    "subject to",
    "depending on",
    "may require",
    "if applicable",
    "where applicable",
    "manager discretion",
    "hiring manager review",
    "founder review",
    "funding dependent",
    "open to international applicants but",
)

SKILL_CLUSTER_ALIASES: Dict[str, Dict[str, Any]] = {
    "relational_database": {
        "members": ["sql", "sqlite", "postgresql", "mysql", "oracle", "relational database", "relational databases"],
        "relation": SkillRelation.RELATED,
    },
    "python": {"members": ["python", "python3"], "relation": SkillRelation.DIRECT},
    "ml_framework": {"members": ["tensorflow", "pytorch", "jax", "keras"], "relation": SkillRelation.RELATED},
    "browser_automation": {"members": ["selenium", "playwright", "cypress", "puppeteer"], "relation": SkillRelation.RELATED},
    "low_level_systems": {"members": ["c", "c++", "rust"], "relation": SkillRelation.PARTIAL},
    "backend_framework": {"members": ["fastapi", "django", "flask", "spring", "express"], "relation": SkillRelation.RELATED},
    "cloud": {"members": ["aws", "gcp", "azure"], "relation": SkillRelation.RELATED},
    "frontend": {"members": ["react", "vue", "angular", "html/css"], "relation": SkillRelation.RELATED},
}

SKILL_RELATIONSHIPS: Dict[str, Dict[str, SkillRelation]] = {
    "tensorflow": {"pytorch": SkillRelation.RELATED, "jax": SkillRelation.RELATED, "keras": SkillRelation.RELATED},
    "pytorch": {"tensorflow": SkillRelation.RELATED, "jax": SkillRelation.RELATED, "keras": SkillRelation.RELATED},
    "selenium": {"playwright": SkillRelation.RELATED, "cypress": SkillRelation.RELATED, "puppeteer": SkillRelation.RELATED},
    "playwright": {"selenium": SkillRelation.RELATED, "cypress": SkillRelation.RELATED, "puppeteer": SkillRelation.RELATED},
    "c++": {"c": SkillRelation.PARTIAL, "rust": SkillRelation.RELATED},
    "c": {"c++": SkillRelation.PARTIAL, "rust": SkillRelation.RELATED},
    "postgresql": {"sqlite": SkillRelation.RELATED, "mysql": SkillRelation.RELATED, "sql": SkillRelation.RELATED},
    "aws": {"gcp": SkillRelation.RELATED, "azure": SkillRelation.RELATED},
    "react": {"vue": SkillRelation.RELATED, "angular": SkillRelation.RELATED, "html/css": SkillRelation.PARTIAL},
    "fastapi": {"django": SkillRelation.RELATED, "flask": SkillRelation.RELATED},
}


def safe_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _token_matches(token: str, text: str) -> bool:
    token = _norm_text(token)
    text = _norm_text(text)
    if not token or not text:
        return False
    if token == text:
        return True
    escaped = re.escape(token)
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", text))


def _clean_skill_target(text: Any) -> str:
    target = _norm_text(text)
    target = re.sub(r"\b(?:experience|fundamentals?|proficiency|proficient|required|preferred|knowledge of|hands-on|strong)\b", " ", target)
    target = re.sub(r"\s+", " ", target).strip(" ,:-/")
    return target


def _split_or_terms(text: str) -> List[str]:
    """Extract skill-like alternatives from an OR requirement."""
    cleaned = re.sub(r"\([^)]*", "(", text)
    pieces = re.split(r"\bor\b|[/|]", cleaned, flags=re.IGNORECASE)
    terms: List[str] = []
    for piece in pieces:
        p = piece.strip(" .,:;()")
        p = re.sub(r"^(?:equivalent|acceptable|modern)\s+", "", p, flags=re.I)
        p = re.sub(r"\b(?:required|preferred|proficiency|proficient|experience|framework|systems programming)\b", "", p, flags=re.I)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            terms.append(p)
    # De-duplicate while preserving order.
    seen = set()
    return [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]


def _infer_education_target(lower: str) -> Optional[Dict[str, Any]]:
    levels: List[str] = []
    for level, patterns in EDUCATION_ALIASES.items():
        if any(re.search(pattern, lower) for pattern in patterns):
            levels.append(level)

    if "technical degree" in lower or "technical discipline" in lower:
        levels = levels or ["bachelor", "master", "phd"]

    domain_terms = []
    domain_map = {
        "computer science": ["computer science", "computing", "cs"],
        "artificial intelligence": ["artificial intelligence", "ai"],
        "machine learning": ["machine learning", "ml"],
        "nursing": ["nursing"],
        "medical": ["medical", "medicine", "medical licensure", "licensure"],
        "electronics": ["electronics", "electrical"],
    }
    for canonical, terms in domain_map.items():
        if any(_token_matches(term, lower) for term in terms):
            domain_terms.append(canonical)

    if levels or domain_terms or "degree in" in lower or "degree" in lower or "technical degree" in lower:
        return {"levels": levels or ["any_degree"], "domains": domain_terms}
    return None


def parse_raw_requirement(
    req_text: str,
    is_explicit_preferred: bool = False,
    is_explicit_hard: bool = False,
) -> Requirement:
    """Convert arbitrary requirement text into a structured requirement."""
    raw = str(req_text or "").strip()
    lower = _norm_text(raw)

    if any(phrase in lower for phrase in CONDITIONAL_PHRASES):
        req_type = RequirementType.CONDITIONAL
    elif is_explicit_preferred or any(word in lower for word in ("preferred", "plus", "nice to have", "bonus", "optional")):
        req_type = RequirementType.PREFERRED
    elif is_explicit_hard:
        req_type = RequirementType.HARD
    else:
        req_type = RequirementType.HARD

    if any(k in lower for k in ("background check", "background & integrity check", "integrity check", "security clearance")):
        return Requirement(raw, RequirementCategory.BACKGROUND_CHECK, req_type, "==", "PASS")

    # Compound enrollment + education/graduation conditions.
    enrollment_markers = (
        "currently enrolled", "currently-enrolled", "enrolled student", "enrollment status",
        "enrolled in", "must have already graduated", "already graduated", "no students",
        "student not eligible", "students not eligible",
    )
    if any(marker in lower for marker in enrollment_markers):
        if "already graduated" in lower or "must have already graduated" in lower:
            primary_target = "GRADUATED"
        elif "no students" in lower or "student not eligible" in lower or "students not eligible" in lower:
            primary_target = "NOT_CURRENTLY_ENROLLED"
        else:
            primary_target = "CURRENTLY_ENROLLED"

        education = _infer_education_target(lower)
        years = [int(y) for y in re.findall(r"\b20\d{2}\b", lower)]
        if education and not years and ("technical degree" in lower or "b.tech" in lower or "btech" in lower):
            secondary_category = RequirementCategory.EDUCATION
            secondary_target = education
        elif years and ("graduat" in lower or "enrolled student graduation" in lower):
            secondary_category = RequirementCategory.GRADUATION_YEAR
            secondary_target = years
        else:
            secondary_category = RequirementCategory.EDUCATION if education else None
            secondary_target = education

        return Requirement(
            raw_text=raw,
            category=RequirementCategory.ENROLLMENT_STATUS,
            req_type=req_type,
            operator="==",
            target_value=primary_target,
            secondary_category=secondary_category,
            secondary_target=secondary_target,
            secondary_operator="IN" if secondary_category == RequirementCategory.GRADUATION_YEAR else "==" if secondary_category else None,
        )

    # Graduation year windows.
    years = [int(y) for y in re.findall(r"\b20\d{2}\b", lower)]
    if years and ("graduat" in lower or "class of" in lower or "cohort" in lower):
        return Requirement(raw, RequirementCategory.GRADUATION_YEAR, req_type, "IN", sorted(set(years)))

    # CGPA / GPA.
    if any(k in lower for k in ("cgpa", "gpa", "cut-off", "cutoff", "grade equivalent")):
        if ">=" in lower:
            op = ">="
        elif "<=" in lower:
            op = "<="
        elif ">" in lower:
            op = ">"
        elif "<" in lower:
            op = "<"
        else:
            op = ">="
        nums = re.findall(r"\d+(?:\.\d+)?", lower)
        value = float(nums[0]) if nums else None
        return Requirement(raw, RequirementCategory.CGPA, req_type, op, value)

    # Work authorization.
    if any(k in lower for k in ("work authorization", "authorized to work", "visa sponsorship")):
        return Requirement(raw, RequirementCategory.WORK_AUTH, req_type, "==", "Yes")

    # Experience OR student projects.
    if re.search(r"\byears?\b", lower) and re.search(r"\bor\b", lower) and any(k in lower for k in ("project", "portfolio")):
        range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years?", lower)
        if range_match:
            experience_target: Any = (int(range_match.group(1)), int(range_match.group(2)))
        else:
            plus_match = re.search(r"(\d+)\s*\+?\s*years?", lower)
            experience_target = int(plus_match.group(1)) if plus_match else 1
        return Requirement(raw, RequirementCategory.EXPERIENCE, req_type, "OR_PROJECT", experience_target, secondary_category=RequirementCategory.GENERAL, secondary_target="PROJECT_EVIDENCE", secondary_operator="==")

    # Generic experience, with an optional skill dimension such as
    # "3+ Years Python / C++ Experience".
    if re.search(r"\byears?\b", lower):
        range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years?", lower)
        if range_match:
            operator = "BETWEEN"
            target = (int(range_match.group(1)), int(range_match.group(2)))
        else:
            plus_match = re.search(r"(\d+)\s*\+?\s*years?", lower)
            operator = ">=" if plus_match else ">="
            target = int(plus_match.group(1)) if plus_match else 1
        skill_dimension = None
        if "experience" in lower and ("python" in lower or "c++" in lower or "c /" in lower or "c/" in lower):
            candidates = []
            for candidate in ("python", "c++", "rust", "c"):
                if candidate in lower:
                    candidates.append(candidate)
            if candidates:
                skill_dimension = candidates
        return Requirement(raw, RequirementCategory.EXPERIENCE, req_type, operator, target,
                           secondary_category=RequirementCategory.SKILL if skill_dimension else None,
                           secondary_target=skill_dimension,
                           secondary_operator="OR" if skill_dimension else None)

    # Education level/domain requirements.
    edu_target = _infer_education_target(lower)
    if edu_target:
        return Requirement(raw, RequirementCategory.EDUCATION, req_type, "MATCH", edu_target)

    # Skills with explicit OR/equivalent language.
    if " or " in lower or "/" in lower:
        terms = _split_or_terms(lower)
        skillish = not any(k in lower for k in ("years", "degree", "enrolled", "graduat"))
        if skillish and len(terms) >= 2:
            return Requirement(raw, RequirementCategory.SKILL, req_type, "OR", terms)

    # Project requirements.
    if "project" in lower or "portfolio" in lower:
        return Requirement(raw, RequirementCategory.GENERAL, req_type, "PROJECT_EVIDENCE", True)

    # Generic skill requirement.
    return Requirement(raw, RequirementCategory.SKILL, req_type, "in", lower)


def _evidence(source: str, field: str, value: Any, confidence: float = 1.0) -> List[Evidence]:
    return [Evidence(source=source, field=field, value=value, confidence=confidence)]


def _uncertain(req: Requirement, is_hard: bool, reason: str) -> RequirementEvaluation:
    return RequirementEvaluation(req, "UNCERTAIN", SkillRelation.UNKNOWN, blocking=is_hard, reason=reason, confidence=0.5)


def _fail(req: Requirement, is_hard: bool, reason: str, relation: SkillRelation = SkillRelation.UNKNOWN) -> RequirementEvaluation:
    return RequirementEvaluation(req, "FAIL", relation, blocking=is_hard, reason=reason)


def _pass(req: Requirement, is_hard: bool, reason: str, relation: SkillRelation = SkillRelation.DIRECT, evidence: Optional[List[Evidence]] = None, sub_paths: Optional[List[SubPathEvaluation]] = None) -> RequirementEvaluation:
    return RequirementEvaluation(req, "PASS", relation, blocking=is_hard, reason=reason, evidence=evidence or [], sub_paths=sub_paths or [])


def _profile_has_bachelor_level(user_profile: Dict[str, Any]) -> bool:
    """Return True when the profile contains credible undergraduate-level evidence.

    The UI labels the degree field "Current / Highest Degree", so compact values such as
    "AIML" are valid user input. For these branch-only values, current enrollment or a
    future graduation year is used as supporting evidence for bachelor-level matching.
    """
    degree = _norm_text(user_profile.get("degree"))
    if not degree:
        return False

    explicit_bachelor = any(re.search(pattern, degree) for pattern in EDUCATION_ALIASES["bachelor"])
    if explicit_bachelor:
        return True

    compact_branch = any(
        alias in degree
        for alias in ("aiml", "ai/ml", "ai & ml", "ai and ml", "artificial intelligence and machine learning")
    )
    if not compact_branch:
        return False

    enrollment = _norm_text(user_profile.get("enrollment_status"))
    grad_year = safe_int(user_profile.get("graduation_year"))
    return enrollment == "currently_enrolled" or (grad_year is not None and grad_year >= 2025)


def _profile_domain_matches(user_profile: Dict[str, Any], domains: Sequence[str]) -> bool:
    """Match compact degree/branch names against requested education domains."""
    degree = _norm_text(user_profile.get("degree"))
    if not degree:
        return False
    for domain in domains:
        aliases = PROFILE_DEGREE_DOMAIN_ALIASES.get(domain, [domain])
        if any(alias in degree for alias in aliases):
            return True
    return False


def _education_matches(profile_degree: str, target: Dict[str, Any]) -> bool:
    degree = _norm_text(profile_degree)
    if not degree:
        return False

    levels = target.get("levels", ["any_degree"])
    domains = target.get("domains", [])
    if "any_degree" in levels and not domains:
        return True

    level_match = False
    for level in levels:
        if level == "any_degree":
            level_match = True
            break
        if level == "bachelor" and _profile_has_bachelor_level({"degree": profile_degree}):
            level_match = True
            break
        if any(re.search(pattern, degree) for pattern in EDUCATION_ALIASES.get(level, [])):
            level_match = True
            break

    if not level_match:
        # A compact AIML/AI-ML value is a valid branch label, but it needs context
        # from the caller's profile. The profile-degree-only fallback remains strict.
        if "bachelor" in levels and any(alias in degree for alias in ("aiml", "ai/ml", "ai & ml", "ai and ml")):
            level_match = True

    if not level_match:
        return False

    if not domains:
        return True

    domain_aliases = {
        "computer science": ["computer science", "computing", "cse", "cs"],
        "artificial intelligence": ["artificial intelligence", "ai", "aiml", "ai/ml", "ai & ml", "ai and ml"],
        "machine learning": ["machine learning", "ml", "aiml", "ai/ml", "ai & ml", "ai and ml"],
        "nursing": ["nursing"],
        "medical": ["medical", "medicine", "medical licensure", "licensure"],
        "electronics": ["electronics", "electrical"],
    }
    return any(_token_matches(term, degree) or term.replace(" ", "") in degree.replace(" ", "") for domain in domains for term in domain_aliases.get(domain, [domain]))


def _evaluate_skill_alternatives(user_skills: Sequence[str], alternatives: Sequence[str]) -> Tuple[str, SkillRelation, Optional[str], Optional[str]]:
    if not user_skills:
        return "UNCERTAIN", SkillRelation.UNKNOWN, None, "Candidate skills are missing from the knowledge base."

    normalized_skills = [_norm_text(s) for s in user_skills if _norm_text(s)]
    unknown_count = 0
    best_relation = SkillRelation.UNKNOWN
    best_candidate = None
    for alt in alternatives:
        alt_norm = _norm_text(alt)
        for candidate in normalized_skills:
            if _token_matches(alt_norm, candidate):
                return "PASS", SkillRelation.DIRECT, candidate, f"Direct skill match verified for '{alt}'."

            for target, related in SKILL_RELATIONSHIPS.items():
                if _token_matches(target, alt_norm):
                    relation = related.get(candidate)
                    if relation in {SkillRelation.RELATED, SkillRelation.PARTIAL}:
                        best_relation = relation
                        best_candidate = candidate

            for cluster in SKILL_CLUSTER_ALIASES.values():
                members = cluster["members"]
                if any(_token_matches(m, alt_norm) for m in members) and any(_token_matches(m, candidate) for m in members):
                    return "PASS", cluster["relation"], candidate, f"Capability cluster match: '{candidate}' satisfies '{alt}'."

        unknown_count += 1

    if best_candidate:
        return "PASS", best_relation, best_candidate, f"Related capability '{best_candidate}' satisfies the requirement."
    return "FAIL", SkillRelation.UNKNOWN, None, None


def evaluate_atomic_requirement(user_profile: Dict[str, Any], req: Requirement) -> RequirementEvaluation:
    user_profile = user_profile or {}
    user_skills = user_profile.get("skills") or []
    user_degree = str(user_profile.get("degree") or "")
    is_hard = req.req_type == RequirementType.HARD

    if req.category == RequirementCategory.BACKGROUND_CHECK:
        status = _norm_text(user_profile.get("background_check_status"))
        if not status:
            return RequirementEvaluation(req, "UNCERTAIN", SkillRelation.UNKNOWN, blocking=False, reason="Background-check status is not yet recorded; this is treated as non-blocking onboarding uncertainty.")
        if status.upper() == "PASS":
            return _pass(req, is_hard, "Candidate has a recorded background-check PASS status.", evidence=_evidence("user_profile.background_check_status", "background_check_status", status.upper()))
        if status.upper() == "FAIL":
            return _fail(req, is_hard, "Candidate has a recorded background-check FAIL status.", SkillRelation.DIRECT)
        return RequirementEvaluation(req, "UNCERTAIN", SkillRelation.UNKNOWN, blocking=False, reason=f"Background-check status '{status}' is not a recognized terminal value.")

    if req.category == RequirementCategory.CGPA:
        cgpa = safe_float(user_profile.get("cgpa"))
        target = safe_float(req.target_value)
        if cgpa is None:
            return _uncertain(req, is_hard, "Candidate CGPA is missing or malformed in the knowledge base.")
        if target is None:
            return _uncertain(req, is_hard, "Requirement CGPA cutoff could not be parsed safely.")
        passed = {
            ">=": cgpa >= target,
            "<=": cgpa <= target,
            ">": cgpa > target,
            "<": cgpa < target,
        }.get(req.operator, False)
        if passed:
            return _pass(req, is_hard, f"Candidate CGPA {cgpa:g} satisfies {req.operator} {target:g}.", evidence=_evidence("user_profile.cgpa", "cgpa", cgpa))
        return _fail(req, is_hard, f"Candidate CGPA {cgpa:g} does not satisfy {req.operator} {target:g}.", SkillRelation.DIRECT)

    if req.category == RequirementCategory.GRADUATION_YEAR:
        year = safe_int(user_profile.get("graduation_year"))
        if year is None:
            return _uncertain(req, is_hard, "Candidate graduation year is missing or malformed in the knowledge base.")
        targets = req.target_value if isinstance(req.target_value, list) else [req.target_value]
        targets = [safe_int(t) for t in targets if safe_int(t) is not None]
        if not targets:
            return _uncertain(req, is_hard, "Requirement graduation-year window could not be parsed.")
        if year in targets:
            return _pass(req, is_hard, f"Graduation year {year} is within the required years {targets}.", evidence=_evidence("user_profile.graduation_year", "graduation_year", year))
        return _fail(req, is_hard, f"Graduation year {year} is outside the required years {targets}.", SkillRelation.DIRECT)

    if req.category == RequirementCategory.WORK_AUTH:
        value = _norm_text(user_profile.get("work_authorization"))
        if not value or value in {"unknown", "none", "null"}:
            return _uncertain(req, is_hard, "Work authorization is missing or unverified.")
        if value in {"yes", "true", "authorized"}:
            return _pass(req, is_hard, "Candidate has verified work authorization.", evidence=_evidence("user_profile.work_authorization", "work_authorization", user_profile.get("work_authorization")))
        return _fail(req, is_hard, "Candidate is not authorized under the required work-authorization criterion.", SkillRelation.DIRECT)

    if req.category == RequirementCategory.EDUCATION:
        degree = _norm_text(user_degree)
        if not degree:
            return _uncertain(req, is_hard, "Candidate degree is missing from the knowledge base.")
        target = req.target_value if isinstance(req.target_value, dict) else {"levels": [str(req.target_value)], "domains": []}
        if _education_matches(degree, target) or (
            "bachelor" in target.get("levels", [])
            and _profile_has_bachelor_level(user_profile)
            and (not target.get("domains") or _profile_domain_matches(user_profile, target.get("domains", [])))
        ):
            return _pass(req, is_hard, f"Candidate degree '{user_profile.get('degree')}' satisfies the required education profile.", evidence=_evidence("user_profile.degree", "degree", user_profile.get("degree")))
        return _fail(req, is_hard, f"Candidate degree '{user_profile.get('degree')}' does not satisfy the required education profile.", SkillRelation.DIRECT)

    if req.category == RequirementCategory.ENROLLMENT_STATUS:
        current = _norm_text(user_profile.get("enrollment_status")).upper()
        if not current:
            return _uncertain(req, is_hard, "Candidate enrollment status is missing from the knowledge base.")
        target = str(req.target_value or "").upper()
        if target == "NOT_CURRENTLY_ENROLLED":
            primary_pass = current in {"GRADUATED", "NOT_CURRENTLY_ENROLLED", "NO"}
        else:
            primary_pass = current == target

        primary_path = SubPathEvaluation("Enrollment status", "PASS" if primary_pass else "FAIL", f"Candidate status={current}, required={target}.")
        if not primary_pass:
            return RequirementEvaluation(req, "FAIL", SkillRelation.DIRECT, blocking=is_hard, sub_paths=[primary_path], reason=f"Candidate enrollment status ({current}) does not satisfy required status ({target}).")

        if req.secondary_category == RequirementCategory.EDUCATION:
            target_profile = req.secondary_target if isinstance(req.secondary_target, dict) else {"levels": [str(req.secondary_target or "bachelor")], "domains": []}
            degree = _norm_text(user_degree)
            if not degree:
                secondary = SubPathEvaluation("Education", "UNKNOWN", "Candidate degree is missing.")
                return RequirementEvaluation(req, "UNCERTAIN", SkillRelation.UNKNOWN, blocking=is_hard, sub_paths=[primary_path, secondary], reason="Enrollment matches, but the required education evidence is missing.")
            education_pass = _education_matches(degree, target_profile) or (
                "bachelor" in target_profile.get("levels", [])
                and _profile_has_bachelor_level(user_profile)
                and (not target_profile.get("domains") or _profile_domain_matches(user_profile, target_profile.get("domains", [])))
            )
            secondary = SubPathEvaluation("Education", "PASS" if education_pass else "FAIL", f"Candidate degree={user_profile.get('degree')}.")
            if not education_pass:
                return RequirementEvaluation(req, "FAIL", SkillRelation.DIRECT, blocking=is_hard, sub_paths=[primary_path, secondary], reason="Enrollment status matches, but required education does not.")
            return RequirementEvaluation(req, "PASS", SkillRelation.DIRECT, blocking=is_hard, sub_paths=[primary_path, secondary], evidence=_evidence("user_profile.enrollment_status", "enrollment_status", current) + _evidence("user_profile.degree", "degree", user_profile.get("degree")), reason="Enrollment and education dimensions both match.")

        if req.secondary_category == RequirementCategory.GRADUATION_YEAR:
            secondary_req = Requirement(req.raw_text, RequirementCategory.GRADUATION_YEAR, req.req_type, "IN", req.secondary_target)
            secondary = evaluate_atomic_requirement(user_profile, secondary_req)
            sub = [primary_path] + [SubPathEvaluation("Graduation year", secondary.status, secondary.reason)]
            if secondary.status == "UNCERTAIN":
                return RequirementEvaluation(req, "UNCERTAIN", SkillRelation.UNKNOWN, blocking=is_hard, sub_paths=sub, reason="Enrollment matches, but graduation-year evidence is missing or malformed.")
            if secondary.status == "FAIL":
                return RequirementEvaluation(req, "FAIL", SkillRelation.DIRECT, blocking=is_hard, sub_paths=sub, reason=secondary.reason)
            return RequirementEvaluation(req, "PASS", SkillRelation.DIRECT, blocking=is_hard, sub_paths=sub, reason="Enrollment and graduation-year dimensions both match.")

        return _pass(req, is_hard, f"Candidate enrollment status ({current}) satisfies the requirement.", evidence=_evidence("user_profile.enrollment_status", "enrollment_status", current))

    if req.category == RequirementCategory.EXPERIENCE:
        if req.operator == "OR_PROJECT":
            exp_value = safe_int(user_profile.get("years_experience"))
            projects = user_profile.get("projects") or []
            exp_status = "UNKNOWN" if exp_value is None else ("PASS" if (exp_value >= req.target_value[0] if isinstance(req.target_value, tuple) else exp_value >= int(req.target_value)) else "FAIL")
            exp_reason = "Years of experience missing." if exp_value is None else f"Candidate has {exp_value} year(s) of experience."
            project_pass = bool(projects) and any(str(p).strip() for p in projects)
            project_status = "PASS" if project_pass else "UNKNOWN"
            paths = [SubPathEvaluation("Path A: Experience", exp_status, exp_reason), SubPathEvaluation("Path B: Student/portfolio projects", project_status, "Project evidence exists." if project_pass else "No project evidence is recorded.")]
            statuses = ["UNCERTAIN" if p == "UNKNOWN" else p for p in (exp_status, project_status)]
            if "PASS" in statuses:
                return RequirementEvaluation(req, "PASS", SkillRelation.DIRECT if exp_status == "PASS" else SkillRelation.RELATED, blocking=is_hard, sub_paths=paths, reason="Compound OR requirement satisfied.")
            if all(s == "FAIL" for s in statuses):
                return RequirementEvaluation(req, "FAIL", SkillRelation.UNKNOWN, blocking=is_hard, sub_paths=paths, reason="All OR paths failed.")
            return RequirementEvaluation(req, "UNCERTAIN", SkillRelation.UNKNOWN, blocking=is_hard, sub_paths=paths, reason="At least one OR path remains uncertain.")

        exp = safe_int(user_profile.get("years_experience"))
        if exp is None:
            return _uncertain(req, is_hard, "Years of experience is missing or malformed in the knowledge base.")
        if req.operator == "BETWEEN":
            low, high = req.target_value
            passed = low <= exp <= high
        else:
            target = safe_int(req.target_value)
            if target is None:
                return _uncertain(req, is_hard, "Experience requirement could not be parsed.")
            passed = exp >= target
            low, high = target, None
        if not passed:
            detail = f"outside range {low}-{high}" if req.operator == "BETWEEN" else f"below minimum {low}"
            return _fail(req, is_hard, f"Candidate experience {exp} year(s) is {detail}.", SkillRelation.DIRECT)

        # A BETWEEN requirement is complete at this point. Do not fall through to
        # the scalar-target branch, which would incorrectly treat the tuple (0, 1)
        # as an invalid integer and turn a valid match into UNCERTAIN.
        if req.operator == "BETWEEN":
            return _pass(
                req,
                is_hard,
                f"Candidate experience {exp} year(s) is within the required range {low}-{high}.",
                SkillRelation.DIRECT,
                _evidence("user_profile.years_experience", "years_experience", exp),
            )

        if req.secondary_category == RequirementCategory.SKILL and req.secondary_operator == "OR":
            status, relation, candidate, reason = _evaluate_skill_alternatives(user_skills, req.secondary_target or [])
            if status == "UNCERTAIN":
                return _uncertain(req, is_hard, "Experience threshold is met, but the required skill evidence is missing.")
            if status == "FAIL":
                return _fail(req, is_hard, "Experience threshold is met, but none of the required technical skills are verified.")
            return _pass(req, is_hard, f"Experience threshold and technical skill requirement are satisfied ({candidate}).", relation, _evidence("user_profile.years_experience", "years_experience", exp) + _evidence("user_profile.skills", "skills", candidate, 0.9))

        target = safe_int(req.target_value)
        if target is None:
            return _uncertain(req, is_hard, "Experience requirement could not be parsed.")
        return _pass(req, is_hard, f"Candidate experience {exp} year(s) meets minimum {target}.", evidence=_evidence("user_profile.years_experience", "years_experience", exp))

    if req.operator == "PROJECT_EVIDENCE":
        projects = user_profile.get("projects") or []
        if not projects:
            return RequirementEvaluation(req, "FAIL", False, blocking=False, reason="No project or portfolio evidence is recorded for this non-blocking criterion.")
        return _pass(req, is_hard, "Candidate has recorded project/portfolio evidence.", relation=SkillRelation.DIRECT, evidence=_evidence("user_profile.projects", "projects", projects))

    # Skill evaluation.
    if isinstance(req.target_value, list) and req.operator == "OR":
        status, relation, candidate, reason = _evaluate_skill_alternatives(user_skills, req.target_value)
        if status == "PASS":
            return _pass(req, is_hard, reason or "OR skill requirement satisfied.", relation, _evidence("user_profile.skills", "skills", candidate, 0.9))
        if status == "UNCERTAIN":
            return _uncertain(req, is_hard, reason or "Candidate skill evidence is missing.")
        return _fail(req, is_hard, f"None of the accepted skills {req.target_value} are verified in the candidate profile.")

    target = _clean_skill_target(req.target_value)
    if not user_skills:
        return _uncertain(req, is_hard, "Candidate skills are missing from the knowledge base.")

    # Project-like skill requirement can still be satisfied by explicit project descriptions.
    if "project" in target or "portfolio" in target:
        projects = user_profile.get("projects") or []
        if projects:
            return _pass(req, is_hard, "Recorded project evidence satisfies the project-oriented requirement.", evidence=_evidence("user_profile.projects", "projects", projects))

    for candidate in user_skills:
        if _token_matches(target, candidate):
            return _pass(req, is_hard, f"Direct skill match verified: '{candidate}'.", SkillRelation.DIRECT, _evidence("user_profile.skills", "skills", candidate))

    # Alias clusters.
    for cluster in SKILL_CLUSTER_ALIASES.values():
        if any(_token_matches(member, target) for member in cluster["members"]):
            for candidate in user_skills:
                if any(_token_matches(member, candidate) for member in cluster["members"]):
                    return _pass(req, is_hard, f"Capability cluster match: '{candidate}' satisfies '{req.raw_text}'.", cluster["relation"], _evidence("user_profile.skills", "skills", candidate, 0.85))

    # Explicit semantic relations.
    for target_canonical, related in SKILL_RELATIONSHIPS.items():
        if _token_matches(target_canonical, target):
            for candidate, relation in related.items():
                if any(_token_matches(candidate, s) for s in user_skills) and relation in {SkillRelation.RELATED, SkillRelation.PARTIAL}:
                    return _pass(req, is_hard, f"{relation.value} capability bridge: '{candidate}' satisfies '{target_canonical}'.", relation, _evidence("user_profile.skills", "skills", candidate, 0.75))

    return _fail(req, is_hard, f"Skill requirement '{req.raw_text}' is not verified in the candidate profile.")


def _or_truth(statuses: Iterable[str]) -> str:
    statuses = [s.upper() for s in statuses]
    if "PASS" in statuses:
        return "PASS"
    if statuses and all(s == "FAIL" for s in statuses):
        return "FAIL"
    return "UNCERTAIN"


def evaluate_candidate_match(user_profile: Dict[str, Any], opportunity: CanonicalOpportunity) -> MatchAnalysisResult:
    """Evaluate the entire opportunity deterministically."""
    all_raw: List[Tuple[str, RequirementType]] = []
    all_raw.extend((r, RequirementType.HARD) for r in opportunity.hard_requirements)
    all_raw.extend((r, RequirementType.PREFERRED) for r in opportunity.preferred_requirements)
    all_raw.extend((r, RequirementType.CONDITIONAL) for r in opportunity.conditional_requirements)

    existing = {str(text).strip().lower() for text, _ in all_raw}
    for requirement in opportunity.requirements:
        if str(requirement).strip().lower() not in existing:
            all_raw.append((requirement, RequirementType.HARD))

    parsed: List[Requirement] = [
        parse_raw_requirement(text, is_explicit_preferred=(typ == RequirementType.PREFERRED), is_explicit_hard=(typ == RequirementType.HARD))
        for text, typ in all_raw
    ]

    evaluations: List[RequirementEvaluation] = []
    hard_satisfied = hard_total = pref_satisfied = pref_total = cond_total = 0
    missing_hard: List[str] = []
    missing_pref: List[str] = []
    uncertain_criteria: List[str] = []
    conditional_criteria: List[str] = []
    skill_evals: Dict[str, Tuple[SkillRelation, str]] = {}

    for req in parsed:
        if req.req_type == RequirementType.CONDITIONAL:
            cond_total += 1
            ev = RequirementEvaluation(req, "UNCERTAIN", SkillRelation.PARTIAL, blocking=True, reason="Conditional requirement requires external/manual determination.", confidence=0.6)
            conditional_criteria.append(req.raw_text)
        else:
            ev = evaluate_atomic_requirement(user_profile, req)
            if req.req_type == RequirementType.HARD:
                hard_total += 1
                if ev.status == "PASS":
                    hard_satisfied += 1
                elif ev.status == "FAIL" and ev.blocking:
                    missing_hard.append(req.raw_text)
                elif ev.status == "UNCERTAIN" and ev.blocking:
                    uncertain_criteria.append(req.raw_text)
            else:
                pref_total += 1
                if ev.status == "PASS":
                    pref_satisfied += 1
                else:
                    missing_pref.append(req.raw_text)

        evaluations.append(ev)
        skill_evals[req.raw_text] = (ev.relation, ev.reason)

    if missing_hard:
        formal = "NO"
        gap = "CRITICAL"
        risk = "CRITICAL"
        breakdown = f"Blocking hard criteria failed: {', '.join(missing_hard)}"
    elif uncertain_criteria:
        formal = "UNCERTAIN"
        gap = "MODERATE"
        risk = "HIGH"
        breakdown = f"Blocking evidence is missing: {', '.join(uncertain_criteria)}"
    else:
        formal = "YES"
        if missing_pref:
            gap = "LOW" if pref_satisfied >= max(1, pref_total // 2) else "MODERATE"
        else:
            gap = "NONE"
        risk = "LOW" if gap in {"NONE", "LOW"} else "MEDIUM"
        breakdown = f"All blocking hard criteria are satisfied ({hard_satisfied}/{hard_total})."

    total_required = max(1, hard_total + pref_total)
    evidence_coverage = round((hard_satisfied + pref_satisfied) / total_required, 2)

    # Fit score is intentionally independent from the APPLY/REVIEW/SKIP decision.
    # Hard requirements dominate because they determine formal eligibility.
    hard_coverage = hard_satisfied / hard_total if hard_total else 1.0
    pref_coverage = pref_satisfied / pref_total if pref_total else 1.0

    if formal == "YES":
        # All mandatory requirements pass. 100% means all preferred criteria also pass.
        fit = int(round(70 * hard_coverage + 30 * pref_coverage))
    elif formal == "UNCERTAIN":
        # Do not present an uncertain candidate as a near-perfect match.
        fit = int(round(70 * hard_coverage + 30 * pref_coverage))
        fit = min(fit, 79)
    else:
        # Hard failures should materially reduce the score.
        fit = int(round(50 * hard_coverage + 20 * pref_coverage))

    fit = max(0, min(100, fit))

    relation_weights = {
        SkillRelation.DIRECT: 1.0,
        SkillRelation.RELATED: 0.85,
        SkillRelation.PARTIAL: 0.60,
        SkillRelation.UNKNOWN: 0.0,
    }
    passing_relations = [relation_weights.get(ev.relation, 0.0) for ev in evaluations if ev.status == "PASS"]
    avg_relation = sum(passing_relations) / max(1, len(passing_relations))
    uncertainty_penalty = len(uncertain_criteria) / max(1, len(evaluations))
    confidence_raw = 0.30 * evidence_coverage + 0.25 * 0.90 + 0.20 * avg_relation + 0.25 * (1.0 - uncertainty_penalty)
    confidence = round(max(0.0, min(1.0, confidence_raw)), 2)
    label = "HIGH" if confidence >= 0.85 else "MEDIUM" if confidence >= 0.70 else "LOW"

    return MatchAnalysisResult(
        formal_eligibility=formal,
        hard_satisfied=hard_satisfied,
        hard_total=hard_total,
        pref_satisfied=pref_satisfied,
        pref_total=pref_total,
        conditional_total=cond_total,
        fit_percentage=int(fit),
        evidence_coverage=evidence_coverage,
        gap_severity=gap,
        risk_level=risk,
        reasoning_confidence=confidence,
        reasoning_confidence_label=label,
        missing_hard=missing_hard,
        missing_pref=missing_pref,
        uncertain_criteria=uncertain_criteria,
        conditional_criteria=conditional_criteria,
        skill_evaluations=skill_evals,
        evaluations=evaluations,
        breakdown_text=breakdown,
    )
