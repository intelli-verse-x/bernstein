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

from bernstein.core.orchestration.phase_schemas import (
    PhaseSchemaError,
    PhaseValidationError,
    schema_for_phase,
    validate_phase_output,
)

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
        """Return the default spec for *phase*.

        ``output_schema`` is now the strict per-phase schema from
        :mod:`bernstein.core.orchestration.phase_schemas`, not the shared
        scaffold.  Callers that previously relied on the union shape get
        a tighter contract automatically.
        """
        return cls(
            phase=phase,
            model=_DEFAULT_MODEL_BY_PHASE[phase],
            effort=_DEFAULT_EFFORT_BY_PHASE[phase],
            max_tokens=_DEFAULT_MAX_TOKENS_BY_PHASE[phase],
            output_schema=schema_for_phase(phase),
        )

    def render_prompt_contract(self) -> str:
        """Return a fenced JSON block the executor splats into the prompt.

        The block is ready to inject verbatim into the agent prompt — it
        starts and ends with the standard ```` ```json ```` fences so
        downstream renderers do not need to know about the schema shape.
        Sharing the bytes between the validator and the prompt template
        eliminates the schema-drift axis where prompts drifted out of
        sync with what the validator actually enforced.
        """
        body = json.dumps(self.output_schema, indent=2, sort_keys=True, ensure_ascii=False)
        return f"```json\n{body}\n```"


# ---------------------------------------------------------------------------
# Distilled handoff artefact
# ---------------------------------------------------------------------------


# NOTE: the legacy shared schema constant has been retired.  Each phase now
# carries its own contract — see :mod:`bernstein.core.orchestration.phase_schemas`.
# The four-field "research-shape" remains the canonical *minimum* and is what
# :class:`PhaseArtifact` instances default to when constructed without explicit
# extras; per-phase extras (``dependencies``, ``files_changed`` …) live in
# :attr:`PhaseArtifact.extras` and are merged into the serialised payload.


@dataclass
class PhaseArtifact:
    """Distilled handoff between phases.

    The implement phase receives only this structure — never the raw
    transcript of the research/plan phases.  Keep entries terse: the whole
    point is to compress N kilobytes of exploration into a few hundred bytes
    of explicit conclusions.

    Attributes:
        summary: One-paragraph distillation of what was learned/decided.
        decisions: Atomic decisions; phase-specific markers (``<id:foo>``)
            participate in the boundary-gate cross-references.
        constraints: Hard constraints carried forward.
        open_questions: Outstanding questions; the boundary gate seeds
            this list back into the next prompt when validation fails.
        extras: Per-phase extension fields (``dependencies`` for plan,
            ``files_changed`` / ``tests_added`` / ``tests_passing`` for
            implement, ``verdict`` for verify).  Persisted alongside the
            four core fields so :func:`validate_phase_output` sees the
            full payload.
    """

    summary: str
    decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return the dict that gets validated and serialised.

        Extras are merged in last so a malicious caller cannot shadow the
        four core fields by stuffing same-named keys into ``extras``.
        """
        out: dict[str, Any] = dict(self.extras)
        out["summary"] = self.summary
        out["decisions"] = list(self.decisions)
        out["constraints"] = list(self.constraints)
        out["open_questions"] = list(self.open_questions)
        return out

    def to_json(self) -> str:
        """Serialise to JSON for storage and as next-phase prompt input."""
        return json.dumps(self.to_payload(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str, *, phase: Phase | None = None) -> PhaseArtifact:
        """Parse a previously serialised artefact.

        Args:
            raw: JSON body to decode.
            phase: When supplied, the payload is validated against that
                phase's strict schema before construction.  Omit for the
                legacy lenient parse — used by tests and by the
                :class:`ArtifactStore` reader where the phase identity
                has already been encoded into the file path.

        Raises:
            ValueError: If the JSON is malformed.
            PhaseValidationError: If ``phase`` is supplied and the
                payload fails its declared schema.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"phase artefact is not valid JSON: {exc}") from exc
        return cls.from_dict(data, phase=phase)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, phase: Phase | None = None) -> PhaseArtifact:
        """Build a :class:`PhaseArtifact` from a parsed mapping.

        When ``phase`` is supplied the payload is validated by the strict
        per-phase jsonschema validator; otherwise the legacy lenient
        check (just verify the four core keys are present and well-typed)
        is applied so existing test fixtures keep working.

        Raises:
            ValueError: If required core keys are missing or wrong-typed.
            PhaseValidationError: If ``phase`` is supplied and the
                payload fails the strict schema.
        """
        if phase is not None:
            errors = validate_phase_output(phase, data)
            if errors:
                raise PhaseValidationError(
                    phase.value if hasattr(phase, "value") else str(phase),
                    errors,
                )
        else:
            for key in ("summary", "decisions", "constraints", "open_questions"):
                if key not in data:
                    raise ValueError(f"phase artefact missing required key {key!r}")
            if not isinstance(data["summary"], str):
                raise ValueError("'summary' must be a string")
            for k in ("decisions", "constraints", "open_questions"):
                if not isinstance(data[k], list) or not all(isinstance(item, str) for item in data[k]):
                    raise ValueError(f"'{k}' must be a list of strings")

        extras = {k: v for k, v in data.items() if k not in {"summary", "decisions", "constraints", "open_questions"}}
        return cls(
            summary=data["summary"],
            decisions=list(data["decisions"]),
            constraints=list(data["constraints"]),
            open_questions=list(data["open_questions"]),
            extras=extras,
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
        """Return a previously stored artefact, or ``None`` if absent.

        The phase is known from the file layout, so the strict per-phase
        schema is enforced on read — a corrupted artefact on disk surfaces
        as a :class:`PhaseValidationError` rather than as a silent shape
        mismatch a few phases later.
        """
        target = self._task_dir(task_id) / f"{phase.value}.json"
        if not target.exists():
            return None
        return PhaseArtifact.from_json(target.read_text(encoding="utf-8"), phase=phase)

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
            # Strict per-phase schema check: an executor that produced a
            # legally-shaped four-field artefact for, say, ``implement``
            # without the required ``files_changed`` / ``tests_*`` extras
            # is rejected at this boundary rather than slipping through.
            schema_errors = validate_phase_output(phase, artifact.to_payload())
            if schema_errors:
                raise PhaseValidationError(phase.value, schema_errors)
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
    "PhaseSchemaError",
    "PhaseSpec",
    "PhaseValidationError",
    "PhasedRunner",
    "default_phases",
    "is_phased",
    "parse_phases",
    "route_for_phase",
    "task_phases",
]
