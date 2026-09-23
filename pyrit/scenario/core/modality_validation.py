# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Derived modality compatibility for scenario attacks.

Modality is not stored on a technique: whether a run is compatible depends on the seed pieces,
the request converters configured for that run, and the concrete target. This module derives
the answer instead.

Two chains are checked, both for turn 0 only:

* the **request chain** — the seed's own data types, projected through the request converters,
  must be exactly one of the objective target's advertised input modality combinations;
* the **response chain** — every data type the target may emit must be one the scorer declares
  it can read.

Anything indeterminate resolves to ``ModalityVerdict.UNKNOWN`` and never blocks a run. A
target whose capabilities cannot be read, an attack that exposes no scoring config, and a
scorer that never declared its data types are all "cannot tell", not "incompatible".

Turns after the first are not modelled here. Media routing across turns belongs to
``_ModalityFeedbackRouter``, which each multi-turn attack consults at execution time.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.executor.attack.core.attack_strategy import AttackStrategy
    from pyrit.models import AttackSeedGroup, AttackTechniqueSeedGroup
    from pyrit.models.literals import PromptDataType
    from pyrit.prompt_normalizer import ConverterConfiguration
    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario.core.atomic_attack import AtomicAttack
    from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class ModalityPolicy(str, Enum):
    """
    Policy for what to do when an atomic attack's modality chain is incompatible.

    Mirrors ``ScorerOverridePolicy`` in vocabulary and meaning, with one difference: ``SKIP``
    logs a warning rather than staying silent, because dropping a whole atomic attack changes
    what a run covers and should be visible even when the user allows it.
    """

    #: Drop the incompatible atomic attack and continue with the rest.
    SKIP = "skip"

    #: Keep the attack and log a warning; it will fail later if the incompatibility is real.
    WARN = "warn"

    #: Abort initialization with ``ModalityValidationError``.
    RAISE = "raise"


class ModalityValidationError(ValueError):
    """
    Raised when an atomic attack's modality chain cannot reach its target or scorer.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers around
    ``initialize_async`` keep working, mirroring ``TechniqueResolutionError``.
    """


class ModalityVerdict(str, Enum):
    """The outcome of checking one atomic attack, or one leg of it."""

    #: Every checked chain can carry its payload end to end.
    COMPATIBLE = "compatible"

    #: At least one chain provably cannot.
    INCOMPATIBLE = "incompatible"

    #: Nothing could be determined — capabilities or declarations were unavailable.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModalityReport:
    """The result of validating one ``AtomicAttack``."""

    #: The attack this report describes, used in the policy's log line and error message.
    atomic_attack_name: str

    #: The overall verdict for the attack.
    verdict: ModalityVerdict

    #: Human-readable explanations, each naming the converter, target, or scorer that failed.
    reasons: tuple[str, ...] = field(default_factory=tuple)

    #: The union of data types the request chain produces across this attack's seed groups.
    projected_request_types: frozenset[PromptDataType] = field(default_factory=frozenset)


def project_request_chain(
    *,
    start_types: Sequence[PromptDataType],
    request_converters: Sequence[ConverterConfiguration],
    piece_indexes_known: bool = True,
) -> tuple[set[PromptDataType], str | None]:
    """
    Project the ordered piece types through a request-converter chain.

    Converter selection is per piece; target compatibility is checked against the resulting
    message-level set of types. Retain piece positions until every configuration has run.
    When the caller cannot know the message's indexes, an indexed configuration conservatively
    retains both the converted and original types.

    Args:
        start_types (Sequence[PromptDataType]): The ordered types of the request's message pieces.
        request_converters (Sequence[ConverterConfiguration]): The request converter chain, in
            application order.
        piece_indexes_known (bool): Whether ``start_types`` includes every actual piece in order.
            Set to ``False`` when projecting a factory without a concrete message.

    Returns:
        tuple[set[PromptDataType], str | None]: The data types that can reach the target, and
        ``None``; or an empty set and a message naming the first converter that could not accept
        the type reaching it.
    """
    piece_types: list[set[PromptDataType]] = [{data_type} for data_type in start_types]

    for configuration in request_converters:
        for index, types in enumerate(piece_types):
            if piece_indexes_known and configuration.indexes_to_apply and index not in configuration.indexes_to_apply:
                continue

            next_types: set[PromptDataType] = set()
            for data_type in sorted(types):
                if (
                    configuration.prompt_data_types_to_apply
                    and data_type not in configuration.prompt_data_types_to_apply
                ):
                    next_types.add(data_type)
                    continue

                converted_types: set[PromptDataType] = {data_type}
                for converter in configuration.converters:
                    unsupported = sorted(t for t in converted_types if not converter.input_supported(t))
                    if unsupported:
                        return set(), (
                            f"{type(converter).__name__} does not accept {unsupported}; "
                            f"it accepts {sorted(converter.supported_input_types)}"
                        )
                    converted_types = set(converter.supported_output_types)
                next_types.update(converted_types)

                if not piece_indexes_known and configuration.indexes_to_apply:
                    next_types.add(data_type)
            piece_types[index] = next_types

    output_types: set[PromptDataType] = set()
    for types in piece_types:
        output_types.update(types)
    return output_types, None


def target_accepts(*, target: PromptTarget, request_types: set[PromptDataType]) -> ModalityVerdict:
    """
    Check a projected request against a target's advertised input modality combinations.

    A target advertises the *combinations* of data types it accepts in a single request, and the
    request's own set of data types must be exactly one of them. ``{text, image_path}`` means
    "text with an image", not "an image alone": a target that accepts a lone image also
    advertises ``{image_path}``, as the vision profiles do, while a video-generation target that
    needs a prompt does not. Reading the declarations literally keeps plan-time in step with
    ``_ModalityFeedbackRouter``, which treats a missing bare ``{text}`` combination as "media
    required on every request".

    Args:
        target (PromptTarget): The objective target.
        request_types (set[PromptDataType]): The data types reaching the target.

    Returns:
        ModalityVerdict: ``UNKNOWN`` when the target's capabilities cannot be read, otherwise
        whether ``request_types`` is one of the advertised combinations.
    """
    supported = _read_modalities(target=target, direction="input")
    if supported is None:
        return ModalityVerdict.UNKNOWN
    if not request_types:
        return ModalityVerdict.INCOMPATIBLE
    if frozenset(request_types) in supported:
        return ModalityVerdict.COMPATIBLE
    return ModalityVerdict.INCOMPATIBLE


def scorer_accepts(*, scorer: Scorer | None, target: PromptTarget) -> tuple[ModalityVerdict, str | None]:
    """
    Check what a target may emit against what its scorer declares it can read.

    This is a type-compatibility check only. It establishes that the scorer can *read* the
    response, not that the resulting score is meaningful for the objective.

    Args:
        scorer (Scorer | None): The objective scorer, if the attack exposes one.
        target (PromptTarget): The objective target.

    Returns:
        tuple[ModalityVerdict, str | None]: The verdict, and a reason when incompatible.
    """
    if scorer is None:
        return ModalityVerdict.UNKNOWN, None

    declared = scorer.supported_data_types
    if declared is None:
        return ModalityVerdict.UNKNOWN, None

    output_modalities = _read_modalities(target=target, direction="output")
    if output_modalities is None:
        return ModalityVerdict.UNKNOWN, None

    emitted = {data_type for combination in output_modalities for data_type in combination}
    missing = sorted(emitted - declared)
    if missing:
        return ModalityVerdict.INCOMPATIBLE, (
            f"{type(scorer).__name__} cannot read {missing} which the objective target may emit; "
            f"it declares {sorted(declared)}"
        )
    return ModalityVerdict.COMPATIBLE, None


def validate_atomic_attack(*, atomic_attack: AtomicAttack) -> ModalityReport:
    """
    Derive whether a built ``AtomicAttack`` can carry its payload to its target and scorer.

    Each seed group is projected independently — two groups are two separate requests, so their
    data types are never combined into one. The attack is incompatible if any seed group is, and
    compatible if at least one leg was determinable and none failed.

    Args:
        atomic_attack (AtomicAttack): The attack to check, after construction and before queuing.

    Returns:
        ModalityReport: The verdict and the reasons behind it.
    """
    name = getattr(atomic_attack, "atomic_attack_name", "<unknown>")
    technique = atomic_attack.attack_technique
    attack = technique.attack

    target = attack.get_objective_target()
    request_converters = attack.get_request_converters() or []
    scoring_config = attack.get_attack_scoring_config()
    scorer = getattr(scoring_config, "objective_scorer", None) if scoring_config is not None else None

    reads_next_message = _reads_next_message(attack=attack)
    verdicts: set[ModalityVerdict] = set()
    reasons: list[str] = []
    projected_all: set[PromptDataType] = set()

    for seed_group in _seed_groups_of(atomic_attack):
        start_types = _effective_start_types(
            seed_group=seed_group,
            seed_technique=technique.seed_technique,
            reads_next_message=reads_next_message,
        )
        if start_types is None:
            verdicts.add(ModalityVerdict.UNKNOWN)
            continue

        projected, failure_reason = project_request_chain(
            start_types=start_types, request_converters=request_converters
        )
        if failure_reason is not None:
            verdicts.add(ModalityVerdict.INCOMPATIBLE)
            if failure_reason not in reasons:
                reasons.append(failure_reason)
            continue

        projected_all |= projected
        request_verdict = target_accepts(target=target, request_types=projected)
        verdicts.add(request_verdict)
        if request_verdict is ModalityVerdict.INCOMPATIBLE:
            reason = (
                f"objective target does not accept {sorted(projected)}; "
                f"it accepts {_format_modalities(_read_modalities(target=target, direction='input'))}"
            )
            if reason not in reasons:
                reasons.append(reason)

    scorer_verdict, scorer_reason = scorer_accepts(scorer=scorer, target=target)
    verdicts.add(scorer_verdict)
    if scorer_reason is not None:
        reasons.append(scorer_reason)

    if ModalityVerdict.INCOMPATIBLE in verdicts:
        verdict = ModalityVerdict.INCOMPATIBLE
    elif ModalityVerdict.COMPATIBLE in verdicts:
        # A leg we could not determine does not cancel one we could.
        verdict = ModalityVerdict.COMPATIBLE
    else:
        verdict = ModalityVerdict.UNKNOWN

    return ModalityReport(
        atomic_attack_name=name,
        verdict=verdict,
        reasons=tuple(reasons),
        projected_request_types=frozenset(projected_all),
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _seed_groups_of(atomic_attack: AtomicAttack) -> list[AttackSeedGroup]:
    """Return the attack's seed groups, or an empty list when they cannot be read."""
    try:
        return list(atomic_attack.seed_groups)
    except (AttributeError, TypeError):
        return []


def _reads_next_message(*, attack: AttackStrategy) -> bool:
    """
    Whether the attack builds its first request from the seed's ``next_message``.

    Attacks created via ``AttackParameters.excluding("next_message")`` build turn 0 from the
    objective text instead, so their seed media never reaches the target on the first turn.

    Returns:
        bool: ``True`` when the attack reads ``next_message``, including when it cannot be
        determined (the common case for every standard ``AttackParameters``).
    """
    try:
        return any(f.name == "next_message" for f in dataclasses.fields(attack.params_type))
    except (AttributeError, TypeError):
        return True


def _effective_start_types(
    *,
    seed_group: AttackSeedGroup,
    seed_technique: AttackTechniqueSeedGroup | None,
    reads_next_message: bool,
) -> list[PromptDataType] | None:
    """
    Determine the data types the first request carries for one seed group.

    The technique seed group is merged in first, because a technique's own prompts travel with
    the seed. ``with_technique`` raises for a group whose prompt sequences overlap a simulated
    conversation, so compatibility is checked first and the unmerged group is used otherwise —
    that pairing is rejected elsewhere and is not a modality problem.

    Returns:
        list[PromptDataType] | None: The ordered starting piece types, or ``None`` when undeterminable.
    """
    if not reads_next_message:
        return ["text"]

    group = seed_group
    if seed_technique is not None:
        try:
            if group.is_compatible_with_technique(technique=seed_technique):
                group = group.with_technique(technique=seed_technique)
        except (AttributeError, TypeError, ValueError):
            return None

    try:
        message = group.next_message
    except (AttributeError, TypeError):
        return None

    if message is None:
        # No prompts on the group: the attack sends the objective text.
        return ["text"]

    try:
        types = [piece.converted_value_data_type for piece in message.message_pieces]
    except (AttributeError, TypeError):
        return None

    return types or ["text"]


def _read_modalities(*, target: PromptTarget, direction: str) -> frozenset[frozenset[PromptDataType]] | None:
    """
    Read a target's declared input or output modality combinations.

    Args:
        target (PromptTarget): The target to read.
        direction (str): Either ``"input"`` or ``"output"``.

    Returns:
        frozenset[frozenset[PromptDataType]] | None: The declared combinations, or ``None``
        when the value is not a real ``frozenset`` — most often a test double whose
        capabilities were never configured. Callers treat that as unknown, not as a failure.
    """
    try:
        capabilities = target.configuration.capabilities
        value = capabilities.input_modalities if direction == "input" else capabilities.output_modalities
    except AttributeError:
        return None
    # Statically this is always a frozenset, because ``TargetCapabilities`` validates it, and ty
    # says so. At runtime a test double that never configured capabilities yields a mock instead,
    # and this guard is what turns that into "unknown" rather than an empty combination set that
    # would wrongly read as incompatible. Deliberately redundant, like the defensive checks the
    # Alembic revisions keep.
    return value if isinstance(value, frozenset) else None  # ty: ignore[redundant-condition-strict]


def _format_modalities(modalities: frozenset[frozenset[PromptDataType]] | None) -> str:
    """
    Render modality combinations deterministically for an error message.

    Args:
        modalities (frozenset[frozenset[PromptDataType]] | None): The combinations to render.

    Returns:
        str: A sorted, stable rendering, or ``"<unknown>"``.
    """
    if modalities is None:
        return "<unknown>"
    return str([sorted(combination) for combination in sorted(modalities, key=sorted)])
