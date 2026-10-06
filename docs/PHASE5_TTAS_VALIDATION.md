# Phase 5 TTAS-based Urgency Validation

## Scope

Phase 5 replaces the primary AI-first free-text urgency decision with grounded TTAS evidence plus a deterministic evaluator. It does not change the official Department KB, Department convergence, SQL data, Android, schedule behavior, or doctor scoring.

This implementation is a preliminary screening subset defined by `phase5_ttas_source_package`. It is not a complete clinical TTAS implementation and no live-provider/live-SQL acceptance is claimed here.

## Source Package

The installed knowledge files originate only from the provided package:

- `backend/knowledge/ttas/source_manifest.json`
- `backend/knowledge/ttas/ttas_evidence_schema.json`
- `backend/knowledge/ttas/active_rules.json`
- `backend/knowledge/ttas/old_ailogic_crosswalk.json`

Only rules marked `implementation_status=enabled` execute. `reference_only` rules remain audit data and cannot affect a level candidate.

The runtime copy adds no medical source or threshold. Phase 5 hardening narrows execution where the package cannot support a trustworthy automatic predicate: clinical modifier fields are retained for provenance but their dependent rules are `reference_only`; official complaint codes additionally require explicit structured complaint/region context where that context is available from the official locator.

## Runtime Flow

The existing Turn Interpreter returns one JSON object containing:

1. `semantic_extractions`
2. `pending_answer`
3. `ttas_evidence`

No fixed second TTAS AI call was added. Backend validation enforces the field allow-list, field-specific value type/range, finite confidence, acceptance threshold, semantic status, and current-turn verbatim grounding. AI-provided level, score, warning, or workflow fields are ignored.

AI-first free-text safety turns use this same Turn Interpreter. A safety pending answer adds a validated `answer_assertion` (`present`, `absent`, or `uncertain`). Backend completes a negative safety screen only for a grounded, high-confidence `answered + absent` proposal tied to the opaque `safety_screen` intent. A no-match TTAS result alone never completes the screen. AI-unavailable free text and keyed free-text safety answers fail closed: they do not run a phrase/keyword safety parser, do not complete `red_flags_checked`, and do not infer a TTAS level.

Validated evidence is stored as history and reduced to current field state by latest accepted evidence. The evaluator reads that structured state and trusted age facts only; it never reads raw user text.

## Deterministic Evaluation

- Rule data is loaded lazily and cached.
- Duplicate rule/source IDs, unknown sources, malformed predicates, invalid levels/statuses, and non-HTTPS sources fail closed.
- Missing required evidence remains unknown and is reported in `missing_evidence`.
- Multiple complete matches retain trace and choose the smallest, most urgent level number.
- No complete enabled match returns `insufficient_information` and `level_candidate=null`.
- No path defaults to level 4 or 5.
- Age-dependent rules do not execute unless trusted or grounded age evidence establishes the population.
- `respiratory_distress`, `hemodynamic_status`, `ill_appearing`, `pain_location_class`, `high_risk_injury_mechanism`, and `cardiac_chest_pain_suspected` cannot independently drive an enabled rule where the package lacks an objective deterministic derivation.
- The package supplies no executable SpO2-to-respiratory-modifier threshold, so no threshold was invented; SpO2 alone remains insufficient information.
- E0101 insect-sting rules require `insect_sting_exposure=true` in addition to their criterion evidence.
- T1201/T1206 upper/lower-limb fracture traces require matching `injury_region`; one fact cannot emit both limb codes.
- T010110, T120207, and T120707 remain reference-only because the packaged penetration evidence cannot prove the exact official complaint context.
- A020211 remains reference-only because `tearing_pain` alone cannot prove the A0202 chest-pain/tightness complaint context.
- A040411 remains reference-only because acute visual disturbance alone cannot prove the A0404 headache complaint context.
- T110107 remains reference-only because genital swelling/deformity alone cannot prove the T1101 blunt-trauma complaint context.
- A030712 remains reference-only. The stored crosswalk states that A030712 belongs under A0307 nausea/vomiting while black stool is separately represented under A0310; the current merged evidence field cannot trace those complaints independently.

Each match records `rule_id`, `official_code`, `source_id`, `official_locator`, level, and grounded evidence references.

## Legacy Differences

- The prototype 90/50/20 urgency score and its duration, function, onset, inflammation, treatment, and risk adders were removed; `urgency_score` remains null.
- The old `<40` glucose threshold was not retained. A130409/A130413 preserve the official `<60` criteria and provenance, but are `reference_only` until `hypoglycemia_symptoms` can be established without an unverified AI clinical grouping.
- Blanket high-blood-pressure behavior is not executed because those package rules are `reference_only`.
- A041011/A041017 retain the official six-hour boundary in audit data, but are `reference_only` until stroke-symptom evidence has a Backend-verifiable official definition.
- No-match is insufficient information, not low urgency.

`RED_FLAG_PATTERNS`, safety negation phrase lists, and their free-text urgency normalizer are no longer production medical fallbacks. Safety ordering and checklist state remain Backend-owned workflow, while medical urgency comes only from validated TTAS evidence and the deterministic evaluator.

## Tests

Focused tests cover package integrity, grounding and type validation, unknown vitals, age handling, reference-only exclusion, multiple-match priority, evidence trace, no-match behavior, the single Turn Interpreter safety path, subjective respiratory fail-closed behavior, insect complaint context, upper/lower trauma trace isolation, penetration reference-only behavior, and Phase 4 conversational regressions.
