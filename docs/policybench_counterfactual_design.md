# PromptSec-PolicyBench counterfactual design

## Rationale

A prompt injection is contextual. Counterfactual groups test whether a model responds to the
relevant boundary rather than memorized vocabulary. Siblings use the same or nearly identical
candidate and declare exactly which context variable or presentation changed.

## Group metadata

Every sibling records a `counterfactual_group_id`, `counterfactual_type`, `changed_variable`,
`parent_scenario_id`, `invariant_fields`, and `expected_label_changes`. The metadata is generation
provenance, not a frozen annotation axis. Expected changes are hidden from annotators.

The deterministic validator compares canonical projections, not free-form descriptions. Fields not
declared changeable must be byte-equivalent after canonical serialization. The validator also
requires at least one declared expected annotation change when the family is label-changing. The
sole exception is `CAPABILITY_CHANGE`: its `expected_label_changes` array is intentionally empty
and the validator rejects any invented frozen-label consequence.

## Families

### Policy change

Candidate, goal, source, capabilities, and presentation remain invariant. Only the selected protected
policy or rule boundary changes. Expected changes normally involve protected-policy alignment,
authority, objectives, or verdict only when authority changes accordingly. Alternate policies are
restricted to the same domain and `policy_family`; a policy-change pair cannot create a cross-family
split edge. Source authority is recomputed from the alternate policy rather than inferred from
policy alignment.

### User-goal change

Policy, candidate, source, capabilities, and presentation remain invariant. Only the legitimate user
goal changes. Expected changes primarily test `ALIGNED` versus `MISALIGNED`; policy alignment must
not be changed without rule evidence.

### Source-role change

Policy, goal, candidate, capabilities, and presentation remain invariant. Only the technical source
role changes. Authority or verdict may change when the policy delegates that action to one source
role but not the other.

### Authority-delegation change

Candidate and source role remain invariant. Only the protected policy's delegation boundary changes.
Both siblings use a controlled effective-policy projection whose action rule is identical and whose
explicit `ALLOW_AUTHORITY`/`DENY_AUTHORITY` delegation clause is the only semantic difference. The
catalogue policy ID, policy family, version, catalogue hash, matched action rules, candidate, source,
goal, and capabilities remain invariant. The effective-policy text and checksum are recorded in
provenance; this family never swaps in an unrelated catalogue policy.

### Capability change

Candidate and policy remain invariant. Only available capabilities change. This is deliberately a
label-invariance negative control: the frozen v1.0 taxonomy has no feasibility axis, and tool
availability is neither policy permission nor source authority. Consequently, v0.1 records an empty
`expected_label_changes` array for this family. Relabelling the directive as compliant, conflicting,
authorized, or unauthorized merely because a capability was added or removed would be an unsupported
inference, and both blueprint and realized-pair validators reject it. This is an explicit scientific
limitation and a tension with the goal of label-changing counterfactuals; capability-pair results must
be reported separately rather than made label-changing with an invented taxonomy value.

### Presentation change

Directive semantics remain invariant while linguistic framing changes between operative, quoted or
reported. Exact strings differ only as needed to express framing. Operative labels are recomputed
from the actual policy and source boundary; quoted siblings are not allowed to retain prompt
injection, outside-authority, attack-objective, or detected-verdict state. Forward and reverse pairs
are selected deterministically to avoid category drift. Role-play that seeks current behavior remains
operative.

## Missing delegation context

An `INSUFFICIENT_CONTEXT` blueprint whose missing boundary is authority delegation uses an
action-only effective-policy projection and explicitly omits the source-delegation clause. It does
not expose the full catalogue prose and then merely clear rule IDs. Missing protected-policy or user-
goal boundaries also propagate `authority_status = UNKNOWN`; the pipeline never infers permission
from context that was intentionally removed.

## Leakage controls

All siblings are one atomic split unit. Counterfactual group IDs and semantic duplicate clusters are
globally atomic. Thus, if record A shares a counterfactual group with B and B shares a semantic
duplicate cluster with C, all three receive one split.

Policy, scenario-template, attack-template, and base-generation family IDs are language-stratified
for linkage only. English and French instances of the same raw family may therefore be allocated to
different splits so a true language-OOD view can exist. The split report records that raw-ID reuse in
`raw_leakage_values_by_field`; it is an intentional exception, not evidence that all raw identifiers
are globally leakage-free. For `test_policy_family_ood`, a separate guard excludes every selected raw
policy family from train across languages.

Before materialization, the splitter also checks that transitive grouping has not collapsed an
excessive share into one component, every requested split is populated, full builds have sufficient
independent components per requested split, and allocated targets are within one largest component
of their configured record targets. Held-out domain and language records must be absent from train.

Counterfactual siblings are never used as independent random examples across train and test. A
dedicated `test_counterfactual.jsonl` view evaluates paired consistency; its groups are absent from
training.

## Interpretation

A correct pairwise change is necessary but not sufficient evidence of policy reasoning. Template
artifacts, language regularities, and generated explanations can create shortcuts. Results should
include per-family accuracy and pairwise consistency, and should not be interpreted as validation on
confidential production policies.
