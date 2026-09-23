# PR #2773 reviewer-comment tracker

Source: Copilot review overview and inline comments, plus Roman Lutz's review,
provided by the user. Fact-checked against local HEAD `c6c07bce1` on
`multimodal`; this does not independently verify a newer remote PR head.
The C1 fix is committed; the C2 fix is local. The other four findings remain open.

| ID | Reviewer | Finding | Verdict | Follow-up |
| --- | --- | --- | --- | --- |
| C1 | Copilot | Fully selected `indexes_to_apply` incorrectly preserves the original request type. | **Fixed locally** | Ordered piece projection and regressions for all-selected and partially selected messages. |
| C2 | Copilot | Resume validates unsampled attacks before restoring persisted seed groups. | **Fixed locally** | Replay the stored selection before checking modalities; cover sampled and legacy resume. |
| R1 | Roman Lutz | Composite scorer intersects child modalities although non-applicable children can return `[]`. | **Confirmed** | Reflect actual child applicability, including disjoint modalities; test a real score. |
| R2 | Roman Lutz | Response check requires every target output type, unlike selective message scoring. | **Confirmed** | Compare per-response combinations with the scorer's actual filtering/strictness policy. |
| R3 | Roman Lutz | Response check omits configured response converters. | **Confirmed** | Project the response converter chain before scoring, or report `UNKNOWN` if indeterminate. |
| R4 | Roman Lutz | A wrapper without `next_message` is assumed to send text. | **Confirmed** | Inspect the actual child contract or return `UNKNOWN`; cover a media-seeded sequential attack. |

The review overview's phrase "compound attacks" refers to R1 and R4; its
request for a fresh Copilot review is a workflow suggestion, not another finding.

## Evidence and scope

### C1 - fully selected indexes

**The distinction:** conversion selects and changes **individual pieces by
index** (`pyrit/prompt_normalizer/prompt_normalizer.py:279-288`), but targets
declare compatibility with the **set of types in the complete message**.
The old projection discarded positions too early: a one-piece text message
and a two-piece text message both looked like `{"text"}`. It then assumed
any nonempty index selection left an unconverted type, even when `[0]`
selected the only piece.

The local fix feeds the ordered first-message pieces into
`project_request_chain`, preserves their positions across converter
configurations, and combines the resulting types only before checking the
target. `[0]` on one text piece produces `{"image_path"}`; `[0]` on two
text pieces produces `{"image_path", "text"}`; `[0, 1]` on two produces
`{"image_path"}`. Factory-level converter checks have no concrete message
and deliberately retain their previous conservative branch projection via
`piece_indexes_known=False`. Runtime converter behavior is unchanged.

### C2 - resumed plan

The original ordering applied modality policy to all rebuilt seed groups
*before* narrowing to the persisted plan, so an unsampled incompatible group
could discard an atomic attack needed by a compatible saved group. The local
change retains full-dataset resolution without random resampling, but
reconstructs the stored seed groups via `_apply_persisted_run_plan` (or
`_apply_persisted_objectives` for a legacy result) **before** modality
validation. On resume, default `SKIP` raises `ModalityValidationError` if a
saved group is incompatible: silently removing it would change the run being
resumed. `WARN` retains it, and new-run filtering is unchanged. Legacy run
plan metadata is only written after validation succeeds. Regression tests
exercise both resume formats with incompatible unsampled and saved groups.

### R1 - composite scorer

`TrueFalseCompositeScorer.supported_data_types` intersects every child's
declaration (`pyrit/score/true_false/true_false_composite_scorer.py:75-93`).
At runtime it filters out non-applicable child results before aggregating
(`true_false_composite_scorer.py:145-174`). A message scorer with no supported
piece returns `[]` rather than a negative score (`pyrit/score/message_scorer.py:1150-1170`,
`pyrit/score/true_false/true_false_scorer.py:170-185`); an existing test confirms
the composite ignores a non-applicable child, albeit via role filtering rather
than disjoint modalities
(`tests/unit/score/test_true_false_composite_scorer.py:215-225`).
Text-only and image-only children report an empty intersection even though the
text child can score a text response. The plan-time check rejects the empty
declaration (`modality_validation.py:214-229`). **Regression:** disjoint text
and image children, both the declared capability and an actual text score.
Keep `None` (undeclared child capability) distinct from an empty declared set.

### R2 - selective response scoring

The plan-time check flattens all target output combinations into a union and
requires every emitted type to be declared by the scorer
(`pyrit/scenario/core/modality_validation.py:197-229`). The default
`ScorerPromptValidator` does **not** require every piece to be valid
(`pyrit/score/scorer_prompt_validator.py:26-34,105-130`); `MessageScorer`
filters unsupported pieces before scoring
(`pyrit/score/message_scorer.py:1150-1170,1264-1269`). For a response that
contains both text and audio, `SubStringScorer` can read the text
(`pyrit/score/true_false/substring_scorer.py:23,64-84`), yet plan-time marks
the pair incompatible. The reviewer is right about this case; merely
flattening alternatives also loses whether the target might emit **audio
alone**, which a text scorer cannot meaningfully score. **Regression:** mixed
text/audio response with a default text scorer and the corresponding
`enforce_all_pieces_valid=True` validator; separately assess audio-only
output rather than declaring every target/scorer pairing compatible.

### R3 - response converters

`validate_atomic_attack` passes the raw target straight to `scorer_accepts`
(`pyrit/scenario/core/modality_validation.py:244-291`); the latter only
examines declared target output and scorer types (`modality_validation.py:197-229`).
But `PromptSendingAttack` configures response converters and passes them to
`PromptNormalizer` (`pyrit/executor/attack/single_turn/prompt_sending.py:95-103,325-339`).
The normalizer converts the **last** returned response before returning it
to the attack for scoring (`pyrit/prompt_normalizer/prompt_normalizer.py:186-210`,
`pyrit/executor/attack/single_turn/prompt_sending.py:341-375`). An
audio-to-text converter advertises `audio_path` input and `text` output
(`pyrit/converter/azure_speech_audio_to_text_converter.py:22-38`), illustrating
the reviewer example without needing a live Azure service. **Regression:**
offline audio-to-text conversion with an audio-output target and text scorer;
ensure opaque/conditional/indexed conversions produce `UNKNOWN` where their
final type cannot safely be inferred. The current validation has no explicit
response-converter accessor on `AttackStrategy`.

### R4 - sequential wrapper

`_reads_next_message` infers whether the first request is text solely from the
presence of the `next_message` parameter, and `_effective_start_types` returns
`{"text"}` whenever it is absent
(`pyrit/scenario/core/modality_validation.py:322-357`).
`SequentialAttack` deliberately excludes that parameter **because its child
attacks own their own seed groups**; it dispatches each child and its seed to
`AttackExecutor` (`pyrit/executor/attack/compound/sequential_attack.py:217-237,248-259,289-335`).
`AdaptiveTechniqueDispatcher` builds real `SequentialChildAttack` instances
from seed groups (`pyrit/scenario/scenarios/adaptive/dispatcher.py:212-234`),
and `AdaptiveScenario` wraps those attacks in `AtomicAttack`
(`pyrit/scenario/scenarios/adaptive/adaptive_scenario.py:466-495`).
Consequently a media-seeded child can target an image-only endpoint while the
outer wrapper is falsely projected as sending text. **Regression:** actual
media-seeded `SequentialChildAttack` and image-only child target; the compound
must not be rejected by the wrapper's invented request type. A wrapper may
also have child targets different from its nominal target, so projecting only
against that nominal target is not sufficient.

## Verification performed

The initial fact-check ran four existing tests (4 passed, 70 deselected):
`test_project_request_chain_indexes_to_apply_keeps_both_branches`,
`test_scorer_accepts_missing_emitted_type_is_incompatible`,
`test_composite_scorer_ignores_non_applicable_child`, and
`test_skip_excludes_dropped_attack_from_persisted_run_plan`. The C1 change
adds all-selected, partially selected, chained-index, unknown-index, and
atomic-attack regression coverage. The C2 change adds five resume regression
cases: plan and legacy replay ignore incompatible unsampled groups, both
fail explicitly for incompatible saved groups, and `WARN` retains the saved
selection. The scenario core suite passes (543 tests); targeted Ruff and
`ty` checks pass. The four remaining findings are not fixed.
