# PR #2773 reviewer-comment tracker

Source: Copilot review overview and inline comments, plus Roman Lutz's review,
provided by the user. Fact-checked against local HEAD `c6c07bce1` on
`multimodal`; this does not independently verify a newer remote PR head.
The C1, C2, R1, R2, R3, and R4 fixes are committed. R3 and R4 share a
regression-test helper and were committed together as `d2cea7ed1`.

| ID | Reviewer | Finding | Verdict | Follow-up |
| --- | --- | --- | --- | --- |
| C1 | Copilot | Fully selected `indexes_to_apply` incorrectly preserves the original request type. | **Committed (`0e9aa2dbe`)** | Ordered piece projection and regressions for all-selected and partially selected messages. |
| C2 | Copilot | Resume validates unsampled attacks before restoring persisted seed groups. | **Committed (`ea6a2d24e`)** | Replay the stored selection before checking modalities; cover sampled and legacy resume. |
| R1 | Roman Lutz | Composite scorer intersects child modalities although non-applicable children can return `[]`. | **Committed (`96aed6786`)** | Union known skippable child modalities; preserve `UNKNOWN` for undeclared or strict children. |
| R2 | Roman Lutz | Response check requires every target output type, unlike selective message scoring. | **Committed (`1c14fad68`)** | Check each output combination against scorer strictness; report `UNKNOWN` for partially scorable alternatives. |
| R3 | Roman Lutz | Response check omits configured response converters. | **Committed (`d2cea7ed1`)** | Project response converter declarations; report `UNKNOWN` for indexed or unsupported projection. |
| R4 | Roman Lutz | A wrapper without `next_message` is assumed to send text. | **Committed (`d2cea7ed1`)** | Treat `SequentialAttack` as `UNKNOWN`; cover actual media-seeded child execution. |

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

Runtime filters out children that return `[]` before aggregating, even with
AND (`pyrit/score/true_false/true_false_composite_scorer.py:145-174`).
The old intersection declared no types for a text-only child paired with an
image-only child, incorrectly rejecting a working text response. The local
fix unions declared modalities only when every child is known to skip
unsupported pieces. `ScorerPromptValidator` exposes that behavior from its
`enforce_all_pieces_valid` and `raise_on_no_valid_pieces` settings;
`MessageScorer` and the nested wrappers forward it. Undeclared or strict
children yield `None` (`UNKNOWN`) rather than falsely promising
compatibility. New regression coverage scores actual text through disjoint
children with **both OR and AND**, checks plan-time compatibility, and
exercises strict, undeclared, and nested children. Runtime aggregation is
unchanged.

### R2 - selective response scoring

The old plan-time check flattened all target output combinations into a
single union and required every type, even though a default message scorer
can filter unsupported pieces. The fix checks **each
combination**: a scorer known to skip unsupported data needs at least one
supported type, while a strict scorer needs every type. If every combination
is scorable, the response-chain verdict is `COMPATIBLE`; if none are, it is
`INCOMPATIBLE`; if some are and others are not, it is `UNKNOWN`. This avoids
rejecting a working mixed text/audio response while preserving strict
validation and not promising that a text scorer can score audio-only output.
The request-chain verdict can still make the overall attack report
`COMPATIBLE` when the response leg is `UNKNOWN`; this is an existing report
aggregation convention and does not cause policy to skip.

R2 regression coverage first demonstrated the old false rejection, then
exercised a real `SubStringScorer` scoring a text/audio message: the text
scores true and the audio piece is ignored. The branch policy matrix checks
`SKIP`, `WARN`, and `RAISE` for text-only, mixed text/audio, strict mixed,
audio-only, and alternative text-or-audio target outputs (15 cases). The
single mixed response is compatible under every policy; strict mixed and
audio-only are rejected under `SKIP`/`RAISE` and retained under `WARN`;
separate text-or-audio outputs yield a response-leg `UNKNOWN` and are kept.
Cached `origin/main` has no modality-policy module, so no comparison of
policy verdicts *on main* exists; attempts to create an isolated main
worktree were denied by the environment. The main `SubStringScorer` source
still declares only text, but its exact runtime matrix was **not run on
main**. Do not claim that the main runtime matrix was verified.

### R3 - response converters

The code and the newer fork head both still forwarded response converters
through `PromptSendingAttack` to `PromptNormalizer`, which converts the
**last** response before scoring; neither exposed them to plan-time
`scorer_accepts`. The regression test ran an offline audio-to-text converter
through the real normalizer, then scored its converted response successfully,
while plan-time validation falsely rejected it before this fix. The local
fix adds an attack response-converter accessor and projects each declared
output combination through type-filtered converter configurations before
testing scorer applicability. Indexed conversion (unknown response piece
positions) and chains that fail projection return `UNKNOWN`, avoiding a
false skip. This does not change runtime conversion or pretend to know
the number and order of pieces in an actual target response.

### R4 - sequential wrapper

`SequentialAttack` excludes `next_message` because its children each own
their seed group, target, converters and scorer; the old wrapper check
fabricated a text request and checked the nominal target. The new test
**executes** a real `SequentialChildAttack` with an image seed and image-only
target: the child succeeds, but pre-fix plan-time validation rejected its
wrapper. The local fix returns `UNKNOWN` for the wrapper instead of
misrepresenting its children's requests. That preserves scenario coverage
without claiming child-level validation; a future explicit per-child
first-request contract could add that precision. Direct attacks retain
their existing checks.

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
`ty` checks pass. For R1, the scorer suite plus the affected scenario
modality and policy tests pass (2,447 tests); targeted Ruff and `ty` checks
also pass. The subsequent R2-R4 work is described below.

Updated verification for R2-R4: scenario-core, sequential-attack, and
prompt-sending tests pass together (686 passed). Targeted Ruff, formatter,
`ty`, and `git diff --check` pass. The R3 audio-to-text regression first
failed because plan-time saw raw audio, then passed after response projection;
the test also ran the normalizer conversion and actual text score. R4 first
ran a real image-seeded child successfully, then demonstrated a failing
plan-time verdict; after the wrapper fix it returns `UNKNOWN` without
altering child execution. Main-only baseline runtime testing remains blocked
by denied worktree creation, as noted under R2. Staging and commits were denied temporarily while the user was unavailable;
staging succeeded after the user resumed. An isolated main worktree remained
unavailable, so the `main`-only runtime matrix was not executed; see R2.
