# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Derived modality compatibility for scenario attacks.

Modality is not stored on a technique: whether a run is compatible depends on the seed pieces,
the request converters configured for that run, and the concrete target. This module derives
the answer instead, by projecting the data types a request starts with through the converter
chain and reporting what can reach the target.

``project_request_chain`` is the shared core. ``AttackTechniqueFactory.can_append_request_converter``
uses it to decide whether another converter may be appended, and scenario-level plan-time
validation uses it to decide whether a built ``AtomicAttack`` can run at all. Keeping one
implementation means the two cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pyrit.models.literals import PromptDataType
    from pyrit.prompt_normalizer import ConverterConfiguration


def project_request_chain(
    *,
    start_types: Iterable[PromptDataType],
    request_converters: Sequence[ConverterConfiguration],
) -> tuple[set[PromptDataType], str | None]:
    """
    Project ``start_types`` through a request-converter chain.

    Each configuration is applied in order. A configuration that declares
    ``prompt_data_types_to_apply`` only converts the types it lists; any other type passes
    through unchanged. A configuration with ``indexes_to_apply`` converts only some pieces, so
    the unconverted type survives alongside the converted one and both can reach the target.

    Args:
        start_types (Iterable[PromptDataType]): The data types the request starts with — the
            pieces of the first message sent to the objective target.
        request_converters (Sequence[ConverterConfiguration]): The request converter chain, in
            application order.

    Returns:
        tuple[set[PromptDataType], str | None]: The data types that can reach the target, and
        ``None``; or an empty set and a message naming the first converter that could not accept
        the type reaching it.
    """
    output_types: set[PromptDataType] = set(start_types)

    for configuration in request_converters:
        next_output_types: set[PromptDataType] = set()

        # Sorted for a deterministic failure message when several start types are in play.
        for output_type in sorted(output_types):
            applies_to_type = (
                not configuration.prompt_data_types_to_apply or output_type in configuration.prompt_data_types_to_apply
            )
            if not applies_to_type:
                next_output_types.add(output_type)
                continue

            converted_types: set[PromptDataType] = {output_type}
            for built_in_converter in configuration.converters:
                unsupported = sorted(
                    data_type for data_type in converted_types if not built_in_converter.input_supported(data_type)
                )
                if unsupported:
                    return set(), (
                        f"{type(built_in_converter).__name__} does not accept {unsupported}; "
                        f"it accepts {sorted(built_in_converter.supported_input_types)}"
                    )
                converted_types = set(built_in_converter.supported_output_types)

            next_output_types.update(converted_types)

            # Partial application leaves some pieces unconverted, so the original type survives.
            if configuration.indexes_to_apply:
                next_output_types.add(output_type)

        output_types = next_output_types

    return output_types, None
