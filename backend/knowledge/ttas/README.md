# TTAS Preliminary Screening Knowledge

This directory contains the versioned Phase 5 rule package used by the Backend TTAS preliminary-screening evaluator.

- `source_manifest.json`: official source registry and version strategy.
- `ttas_evidence_schema.json`: allow-listed AI evidence contract.
- `active_rules.json`: deterministic rules; only records with `implementation_status=enabled` may execute.
- `old_ailogic_crosswalk.json`: audit-only comparison with the retired prototype urgency behavior.

The Backend validates AI evidence structure, confidence, and current-turn verbatim grounding. The evaluator reads only validated evidence and trusted age facts; it never reads raw user text. Missing evidence is not treated as normal, and no-match never defaults to TTAS level 4 or 5.

Runtime hardening may only narrow package execution: rules that depend on unverified clinical modifier classification or cannot prove their exact official complaint context are marked `reference_only`. No replacement threshold or medical criterion is invented. Enabled insect-sting and limb-trauma rules require explicit grounded context fields so their official-code trace cannot cross complaints or body regions.

This is a preliminary screening implementation based on the packaged official sources. It is not a complete clinical TTAS system and does not replace professional triage.
