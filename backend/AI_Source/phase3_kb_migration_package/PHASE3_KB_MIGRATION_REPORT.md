# Phase 3 KB migration research report

Generated: 2026-10-05

## Executive decision

Use `AILogic/dept_keywords.py` as a **seed vocabulary**, not as medical truth. A legacy pair can enter production only when official evidence supports it. Official hospital department wording and APP department wording are intentionally separate: the official wording preserves provenance; APP/SQL `dept_id + childDept` is the canonical runtime identity.

## Classification result

| Class | Count | Production role |
|---|---:|---|
| `direct_routing_evidence` | 229 | Primary candidate retrieval/routing evidence |
| `department_specialty_evidence` | 294 | Supporting candidate evidence, lower authority than direct routing |
| `legacy_synonym` | 18 | Semantic normalization only; never independent medical truth |
| `legacy_unverified` | 222 | Disabled in production |
| **Total** | **763** | 35 legacy departments |

`legacy_unverified` does **not** mean wrong. It only means this research pass did not find enough official evidence to enable the pair safely.

## Canonical department rule

- Medical provenance: retain the official source department label exactly as a source label.
- Runtime identity: use the APP/SQL `dept_id`, `parentDept`, and `childDept`.
- Do not require website name == APP name.
- Do not use fuzzy name matching at runtime. Use a reviewed mapping table.
- Revalidate every `canonical_dept_id` against live 310 active departments before deployment.

Examples:

| Official label | APP/SQL canonical | dept_id |
|---|---|---:|
| 心臟科 / 心臟內科 | 心臟內科 | 1241 |
| 內分泌新陳代謝科 / 內分泌暨新陳代謝科 | 新陳代謝科 | 1237 |
| 過敏免疫風濕科 | 過敏免疫風濕 | 1255 |
| 大腸直腸外科 | 直腸外科 | 1245 |
| 泌尿科 / 泌尿部 | 泌尿外科 | 1314 |
| 復健醫學科 | 復健醫學 | 1265 |
| 耳鼻喉科 + ear evidence | 耳科 | 1333 |
| 耳鼻喉科 + nasal evidence | 鼻科 | 1334 |
| 耳鼻喉科 + throat/voice/swallow evidence | 喉科 | 1335 |

## Why this fixes Phase 4

The current Phase 4 convergence engine already validates patient `source_text`, official `source_id`, live `dept_id`, candidate Top-K, and backend-owned resolved/ambiguous gates. The missing piece was real source-grounded multi-department coverage. The current Taipei VGH patient guide itself supplies multi-candidate routing (for example dizziness, chest pain, abdominal pain, urinary symptoms, joint/back pain), so Phase 4 ambiguity no longer needs synthetic “測試甲科/乙科” knowledge.

## Source policy

1. **Priority 1:** Taipei VGH patient guide symptom→department table.
2. **Priority 1:** Taipei VGH department-specific direct routing table (currently internal medicine).
3. **Priority 2:** Taipei VGH official doctor-specialty / department-service / patient-education pages.
4. **Priority 3:** Taichung VGH official data only for gaps or sub-specialty disambiguation (not to override Taipei direct routing).
5. **AILogic:** seed terms / synonyms only; never a provenance source.

## Per-department migration counts

| Legacy department | APP dept_id | APP childDept | Direct | Specialty | Synonym | Unverified | Total |
|---|---:|---|---:|---:|---:|---:|---:|
| 一般內科 | 1232 | 一般內科 | 22 | 0 | 0 | 1 | 23 |
| 家庭醫學科(一般門診/戒菸) | 1233 | 家庭醫學科(一般門診/戒菸) | 13 | 0 | 3 | 22 | 38 |
| 心臟內科 | 1241 | 心臟內科 | 17 | 0 | 0 | 0 | 17 |
| 感染科 | 1234 | 感染科 | 10 | 0 | 1 | 1 | 12 |
| 新陳代謝科 | 1237 | 新陳代謝科 | 13 | 0 | 6 | 17 | 36 |
| 神經內科 | 1274 | 神經內科 | 7 | 12 | 1 | 3 | 23 |
| 胃腸肝膽科 | 1239 | 胃腸肝膽科 | 20 | 0 | 0 | 0 | 20 |
| 胸腔內科 | 1238 | 胸腔內科 | 4 | 5 | 0 | 0 | 9 |
| 腎臟科 | 1242 | 腎臟科 | 10 | 0 | 0 | 0 | 10 |
| 過敏免疫風濕 | 1255 | 過敏免疫風濕 | 10 | 6 | 0 | 7 | 23 |
| 血液腫瘤科 | 1240 | 血液腫瘤科 | 15 | 14 | 0 | 0 | 29 |
| 腫瘤內科 | 1287 | 腫瘤內科 | 0 | 3 | 0 | 13 | 16 |
| 一般外科 | 1290 | 一般外科 | 4 | 5 | 0 | 0 | 9 |
| 直腸外科 | 1245 | 直腸外科 | 2 | 1 | 0 | 0 | 3 |
| 心臟外科 | 1285 | 心臟外科 | 3 | 10 | 0 | 16 | 29 |
| 胸腔外科 | 1309 | 胸腔外科 | 2 | 8 | 0 | 1 | 11 |
| 外傷兼疝氣及肝膽胃腸外科 | 1310 | 外傷兼疝氣及肝膽胃腸外科 | 1 | 0 | 0 | 0 | 1 |
| 骨科 | 1298 | 一般骨科 | 6 | 6 | 0 | 2 | 14 |
| 神經外科 | 1244 | 神經外科 | 4 | 7 | 0 | 0 | 11 |
| 泌尿外科 | 1314 | 泌尿外科 | 5 | 9 | 0 | 0 | 14 |
| 整形外科 | 1294 | 整形外科 | 1 | 36 | 0 | 7 | 44 |
| 眼科 | 1329 | 眼科 | 5 | 3 | 1 | 1 | 10 |
| 牙科 | 1248 | 牙科 | 3 | 5 | 0 | 4 | 12 |
| 口腔顎面外科 | 1250 | 口腔顎面外科 | 4 | 7 | 0 | 0 | 11 |
| 耳科 | 1333 | 耳科 | 8 | 3 | 1 | 0 | 12 |
| 鼻科 | 1334 | 鼻科 | 4 | 5 | 2 | 5 | 16 |
| 喉科 | 1335 | 喉科 | 5 | 6 | 0 | 0 | 11 |
| 婦產科 | 1319 | 婦產科 | 11 | 7 | 0 | 0 | 18 |
| 兒童內科 | 1320 | 兒童內科 | 0 | 0 | 0 | 4 | 4 |
| 皮膚科 | 1252 | 皮膚科 | 3 | 21 | 1 | 4 | 29 |
| 精神科 | 1251 | 精神科 | 2 | 81 | 2 | 96 | 181 |
| 失智特別門診 | 1340 | 失智特別門診 | 0 | 1 | 0 | 0 | 1 |
| 復健醫學 | 1265 | 復健醫學 | 9 | 0 | 0 | 15 | 24 |
| 疼痛控制科 | 1345 | 疼痛控制科 | 6 | 0 | 0 | 3 | 9 |
| 醫學美容中心 | 1293 | 醫學美容中心 | 0 | 33 | 0 | 0 | 33 |

## Integration constraints for the backend

- Do not change SQL schema.
- Do not invent or web-research medical mappings during integration. Use these reviewed artifacts as input.
- `direct_routing_evidence` and `department_specialty_evidence` must carry `supporting_source_ids`.
- `legacy_synonym` belongs in semantic normalization, not the medical routing table.
- `legacy_unverified` must be ignored by production routing.
- Replace exact-name-only KB→DB reconciliation with a reviewed canonical mapping keyed by `canonical_dept_id`; then validate that ID exists exactly once in the live active Department inventory.
- Generic website labels such as 耳鼻喉科 may require concept-specific APP mapping (耳/鼻/喉), not a single global alias.
- Broad legacy 骨科 currently maps to APP `一般骨科 (1298)` because that convention already exists in the project; keep this mapping explicit and revalidate on 310.

## Files in this package

- `phase3_kb_migration_763.json` — full auditable row-by-row classification.
- `phase3_kb_migration_763.csv` — spreadsheet-friendly version.
- `phase3_department_canonical_mapping.json` — 35 legacy departments → APP/SQL canonical identity plus official source labels.
- `phase3_source_registry.json` — official sources and priority policy.
- `phase3_kb_production_seed.json` — routing evidence, semantic synonyms, and disabled legacy-unverified entries split for backend consumption.

## Important limitations

- This is a **research/migration candidate**, not yet committed to the backend.
- The 310 department snapshot is from the repository; live 310 must revalidate active IDs before production.
- Specialty/patient-education evidence is weaker than direct symptom→department guidance and must not override Priority-1 routing evidence.
- No urgency/TTAS work is included; this remains Phase 3 support for Phase 4 only.
