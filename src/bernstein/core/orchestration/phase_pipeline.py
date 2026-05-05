"""Discrete research/plan/implement phase separation with distilled handoffs.

Implements the *discrete-phase-separation* agentic pattern: instead of one
long-running agent that researches, plans, then implements in a single
context window, this module spawns a fresh short-lived agent per phase and
passes only a structured ``PhaseArtifact`` between them.  Each phase starts
with a clean prompt cache; the implement phase never sees the raw research
transcript, only the distilled summary/decisions/constraints/open-questions.

Opt-in: callers must explicitly invoke :class:`PhasedRunner`.  Steps in a
plan YAML opt into multi-phase execution by declaring
``phases: [research, plan, implement]``; a step without ``phases`` runs as a
single phase via the existing pipeline.  See
:data:`bernstein.core.defaults.PHASE_PIPELINE` for the global enable flag.

Pattern source:
    https://github.com/nibzard/awesome-agentic-patterns/blob/main/patterns/discrete-phase-separation.md
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bernstein.core.tasks.models import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase enum & default specs
# ---------------------------------------------------------------------------


class Phase(StrEnum):
    """Discrete phases in the research → plan → implement → verify pipeline."""

    RESEARCH = "research"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"


# Per-phase default routing.  ``research`` and ``plan`` warrant a high-reasoning
# model because the artefact they produce will be cached as ground truth for
# the cheaper ``implement`` agent.  ``verify`` typically only needs to confirm
# acceptance criteria.
_DEFAULT_MODEL_BY_PHASE: dict[Phase, str] = {
    Phase.RESEARCH: "opus",
    Phase.PLAN: "opus",
    Phase.IMPLEMENT: "sonnet",
    Phase.VERIFY: "sonnet",
}

_DEFAULT_EFFORT_BY_PHASE: dict[Phase, str] = {
    Phase.RESEARCH: "high",
    Phase.PLAN: "high",
    Phase.IMPLEMENT: "normal",
    Phase.VERIFY: "normal",
}

_DEFAULT_MAX_TOKENS_BY_PHASE: dict[Phase, int] = {
    Phase.RESEARCH: 60_000,
    Phase.PLAN: 30_000,
    Phase.IMPLEMENT: 80_000,
    Phase.VERIFY: 20_000,
}


@dataclass(frozen=True)
class PhaseSpec:
    """Configuration for one discrete phase invocation.

    Attributes:
        phase: Which phase this spec describes.
        model: Default model identifier (e.g. ``"opus"``).
        effort: Default effort level (``"high"`` etc).
        max_tokens: Soft cap on the prompt+output budget for this phase.
        output_schema: JSON Schema describing the artefact the agent must
            emit.  Used to validate the handoff before the next phase starts.
    """

    phase: Phase
    model: str
    effort: str
    max_tokens: int
    output_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls, phase: Phase) -> PhaseSpec:
        """Return the default spec for *phase*."""
        return cls(
            phase=phase,
            model=_DEFAULT_MODEL_BY_PHASE[phase],
            effort=_DEFAULT_EFFORT_BY_PHASE[phase],
            max_tokens=_DEFAULT_MAX_TOKENS_BY_PHASE[phase],
            output_schema=_PHASE_ARTIFACT_SCHEMA,
        )


# ---------------------------------------------------------------------------
# Distilled handoff artefact
# ---------------------------------------------------------------------------


_PHASE_ARTIFACT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "decisions", "constraints", "open_questions"],
    "additionalProperties": False,
}


@dataclass
class PhaseArtifact:
    """Distilled handoff between phases.

    The implement phase receives only this structure — never the raw
    transcript of the research/plan phases.  Keep entries terse: the whole
    point is to compress N kilobytes of exploration into a few hundred bytes
    of explicit conclusions.
    """

    summary: str
    decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise to JSON for storage and as next-phase prompt input."""
        return json.dumps(
            {
                "summary": self.summary,
                "decisions": self.decisions,
                "constraints": self.constraints,
                "open_questions": self.open_questions,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> PhaseArtifact:
        """Parse a previously serialised artefact.

        Raises:
            ValueError: If the JSON is malformed or fails schema validation.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"phase artefact is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseArtifact:
        """Build a :class:`PhaseArtifact` from a parsed mapping.

        Raises:
            ValueError: If required keys are missing or have the wrong type.
        """
        for key in ("summary", "decisions", "constraints", "open_questions"):
            if key not in data:
                raise ValueError(f"phase artefact missing required key {key!r}")
        if not isinstance(data["summary"], str):
            raise ValueError("'summary' must be a string")
        for k in ("decisions", "constraints", "open_questions"):
            if not isinstance(data[k], list) or not all(isinstance(item, str) for item in data[k]):
                raise ValueError(f"'{k}' must be a list of strings")
        return cls(
            summary=data["summary"],
            decisions=list(data["decisions"]),
            constraints=list(data["constraints"]),
            open_questions=list(data["open_questions"]),
        )


# ---------------------------------------------------------------------------
# Plan-file vocabulary
# ---------------------------------------------------------------------------


_DEFAULT_PHASES: tuple[Phase, ...] = (Phase.RESEARCH, Phase.PLAN, Phase.IMPLEMENT)


def parse_phases(raw: object) -> list[Phase]:
    """Parse a ``phases:`` value from a plan YAML step.

    Accepts a list of strings naming phases.  Empty/None means "single phase"
    and returns ``[]`` so callers can fall back to the legacy single-agent
    pipeline.

    Raises:
        ValueError: When *raw* is neither None nor a list of valid phase names.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"phases must be a list, got {type(raw).__name__}")
    out: list[Phase] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"phase entry must be a string, got {type(item).__name__}")
        try:
            out.append(Phase(item.lower()))
        except ValueError as exc:
            valid = ", ".join(p.value for p in Phase)
            raise ValueError(f"unknown phase {item!r}; valid phases: {valid}") from exc
    return out


def default_phases() -> list[Phase]:
    """Return the canonical research → plan → implement sequence."""
    return list(_DEFAULT_PHASES)


# ---------------------------------------------------------------------------
# Routing integration
# ---------------------------------------------------------------------------


def route_for_phase(
    phase: Phase,
    *,
    task_model: str | None = None,
    task_effort: str | None = None,
) -> tuple[str, str]:
    """Pick a (model, effort) pair for an in-flight phase.

    Manager-specified overrides on the task win when present.  Otherwise the
    phase's default applies — research/plan get a high-reasoning model,
    implement/verify get a cheaper one.
    """
    model = task_model or _DEFAULT_MODEL_BY_PHASE[phase]
    effort = task_effort or _DEFAULT_EFFORT_BY_PHASE[phase]
    return model, effort


# ---------------------------------------------------------------------------
# Artefact persistence
# ---------------------------------------------------------------------------


_DEFAULT_RUNTIME_ROOT = Path(".sdd/runtime/phase_artifacts")


@dataclass
class ArtifactStore:
    """Filesystem-backed store for distilled handoffs.

    Layout: ``<root>/<task_id>/<phase>.json``.  One subdirectory per task
    keeps cleanup atomic — :meth:`gc_task` deletes the whole tree when the
    parent task closes.
    """

    root: Path = field(default_factory=lambda: _DEFAULT_RUNTIME_ROOT)

    def _task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def write(self, task_id: str, phase: Phase, artifact: PhaseArtifact) -> Path:
        """Persist *artifact* and return the path written."""
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        target = task_dir / f"{phase.value}.json"
        target.write_text(artifact.to_json(), encoding="utf-8")
        return target

    def read(self, task_id: str, phase: Phase) -> PhaseArtifact | None:
        """Return a previously stored artefact, or ``None`` if absent."""
        target = self._task_dir(task_id) / f"{phase.value}.json"
        if not target.exists():
            return None
        return PhaseArtifact.from_json(target.read_text(encoding="utf-8"))

    def gc_task(self, task_id: str) -> bool:
        """Delete all artefacts for *task_id*.  Returns True when something was removed."""
        task_dir = self._task_dir(task_id)
        if not task_dir.exists():
            return False
        shutil.rmtree(task_dir, ignore_errors=True)
        return True


# ---------------------------------------------------------------------------
# Phased runner
# ---------------------------------------------------------------------------


PhaseExecutor = Callable[["Task", "PhaseSpec", PhaseArtifact | None], PhaseArtifact]
"""Pluggable phase-execution callable.

Concrete implementations spawn a CLI agent with the model+effort from
*spec*, feed it the prior artefact (or ``None`` for the first phase), wait
for it to emit a structured artefact, and return it.  The runner itself is
agent-agnostic — pass any callable that satisfies this signature in tests,
batch jobs, or the production spawner integration.
"""


@dataclass
class PhaseResult:
    """Outcome of a single phase invocation."""

    phase: Phase
    spec: PhaseSpec
    artifact: PhaseArtifact
    artifact_path: Path
    input_bytes: int
    output_bytes: int


@dataclass
class PhasedRunner:
    """Drive a task through an ordered list of phases with distilled handoffs.

    Each phase runs in a *fresh* invocation of *executor*.  The runner only
    forwards the previous phase's :class:`PhaseArtifact` as seed context —
    not raw transcripts, tool outputs, or anything else.  This is the whole
    point of the pattern.

    Attributes:
        executor: Callable that runs a single phase.  See :data:`PhaseExecutor`.
        store: Where to persist artefacts.  Defaults to ``.sdd/runtime/phase_artifacts``.
        phases: Optional override of the phase sequence.  Defaults to research → plan → implement.
    """

    executor: PhaseExecutor
    store: ArtifactStore = field(default_factory=ArtifactStore)
    phases: list[Phase] = field(default_factory=default_phases)

    def _spec_for(self, task: Task, phase: Phase) -> PhaseSpec:
        # Manager overrides on the task win, otherwise per-phase defaults apply.
        model, effort = route_for_phase(
            phase,
            task_model=getattr(task, "model", None),
            task_effort=getattr(task, "effort", None),
        )
        base = PhaseSpec.default(phase)
        return PhaseSpec(
            phase=phase,
            model=model,
            effort=effort,
            max_tokens=base.max_tokens,
            output_schema=base.output_schema,
        )

    def run(self, task: Task) -> list[PhaseResult]:
        """Execute *task* through all configured phases.

        Returns one :class:`PhaseResult` per phase.  Each result's
        ``input_bytes`` reflects the size of the prior artefact (the only
        seed context the phase received), enabling the size-budget assertion
        called out in the ticket's acceptance criteria.
        """
        results: list[PhaseResult] = []
        prior: PhaseArtifact | None = None
        for phase in self.phases:
            spec = self._spec_for(task, phase)
            input_bytes = len(prior.to_json().encode("utf-8")) if prior is not None else 0
            artifact = self.executor(task, spec, prior)
            if not isinstance(artifact, PhaseArtifact):
                raise TypeError(
                    f"executor returned {type(artifact).__name__} for phase {phase.value}; expected PhaseArtifact"
                )
            output_bytes = len(artifact.to_json().encode("utf-8"))
            path = self.store.write(task.id, phase, artifact)
            logger.info(
                "phase %s for task %s using %s/%s wrote %d bytes (input %d bytes) to %s",
                phase.value,
                task.id,
                spec.model,
                spec.effort,
                output_bytes,
                input_bytes,
                path,
            )
            results.append(
                PhaseResult(
                    phase=phase,
                    spec=spec,
                    artifact=artifact,
                    artifact_path=path,
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                )
            )
            prior = artifact
        return results


# ---------------------------------------------------------------------------
# Public entry-point helper
# ---------------------------------------------------------------------------


def is_phased(task: Task) -> bool:
    """Return True when *task* opts into phased execution.

    Single source of truth: the task's ``metadata['phases']`` list (set by
    the plan loader from a ``phases:`` step field).  Tasks without that key
    run via the existing single-phase pipeline — back-compat unchanged.
    """
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    raw = metadata.get("phases")
    if not raw:
        return False
    try:
        return len(parse_phases(raw)) > 0
    except ValueError:
        return False


def task_phases(task: Task) -> list[Phase]:
    """Return the phase list declared on *task*, or an empty list when absent."""
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    return parse_phases(metadata.get("phases"))


__all__ = [
    "ArtifactStore",
    "Phase",
    "PhaseArtifact",
    "PhaseExecutor",
    "PhaseResult",
    "PhaseSpec",
    "PhasedRunner",
    "default_phases",
    "is_phased",
    "parse_phases",
    "route_for_phase",
    "task_phases",
]
