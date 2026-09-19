import { z } from "zod";

export const VERSION = "danids-explorer-v1";
const hash = z.string().regex(/^[a-f0-9]{64}$/);
const count = z.number().int().nonnegative();
const percent = z.number().min(0).max(100);
const source = z
  .object({
    path: z.string(),
    sha256: hash,
    hash_mode: z.literal("canonical-lf-utf8"),
  })
  .strict();
const wrap = <T extends z.ZodTypeAny>(data: T) =>
  z
    .object({
      schema_version: z.literal(VERSION),
      sources: z.array(source).nonempty(),
      data,
    })
    .strict();

const pathway = z
  .object({
    id: z.enum(["recognition", "capability", "immediate", "sustained"]),
    title: z.string(),
    numerator: count,
    denominator: count.positive(),
    percent: z.string(),
    unit: z.string(),
    outcome: z.string(),
    population: z.string(),
  })
  .strict()
  .refine((v) => v.numerator <= v.denominator);
const method = z
  .object({
    accepted_updates: count,
    compliance_rate: z.number().min(0).max(1),
    labels_requested: count,
    unsafe_exposure_windows: count,
  })
  .strict();
const domain = z.enum(["U", "T", "C", "B"]);
export const schemas = {
  overview: wrap(
    z
      .object({
        chain: z.array(z.string()),
        limitations: z.array(z.string()),
        ledger: z.record(
          z.enum(["SUPPORTED", "NOT_SUPPORTED", "NOT_TESTABLE", "PARTIAL"]),
        ),
      })
      .strict(),
  ),
  cross_domain: wrap(
    z
      .object({
        domains: z.array(domain).length(4),
        cells: z
          .array(
            z
              .object({
                source: domain,
                target: domain,
                recall: z.string(),
                fpr_budget_ratio: z.string(),
              })
              .strict(),
          )
          .length(16),
        precision: z.string(),
        aggregation: z.string(),
        unavailable: z.array(z.string()),
      })
      .strict(),
  ),
  model_health: wrap(
    z
      .object({
        target_fpr: z.number(),
        recall_loss_tolerance: z.number(),
        confidence: z.string(),
        feature_groups: z.array(z.string()).length(5),
        features: z.array(z.string()).length(28),
        states: z
          .array(
            z
              .object({
                name: z.enum(["SAFE", "UNCERTAIN", "HARMFUL"]),
                rule: z.string(),
              })
              .strict(),
          )
          .length(3),
      })
      .strict(),
  ),
  study4: wrap(
    z
      .object({
        methods: z
          .object({
            STATIC: method,
            ALWAYS_ADAPT: method,
            DANIDS_CORE: method,
            OFFLINE_ORACLE: method,
          })
          .strict(),
      })
      .strict(),
  ),
  rdx: wrap(
    z
      .object({
        pathways: z.array(pathway).length(4),
        core_recognised: count,
        core_harmful: count,
        paired_units: count,
        gate_median_pp: z.number(),
        gate_positive_rotations: z.string(),
        decision: z.string(),
        interpretation: z.string(),
        budgets: z
          .array(
            z
              .object({
                budget: z.string(),
                timing: z.enum(["historical", "prospective"]),
                optimizer_rows: count,
                eligible: count,
                mean_p1: percent,
                median_p1: percent,
                mean_p2: percent,
                median_p2: percent,
              })
              .strict(),
          )
          .length(3),
        contrasts: z
          .array(
            z
              .object({
                contrast: z.string(),
                mean_pp: z.number(),
                median_pp: z.number(),
                positive_rotations: z.string(),
              })
              .strict(),
          )
          .length(3),
        actions: z
          .array(
            z
              .object({ id: z.string(), name: z.string(), detail: z.string() })
              .strict(),
          )
          .length(5),
      })
      .strict(),
  ),
  provenance: wrap(
    z
      .object({
        freeze_tag: z.string(),
        rdx_tag: z.string(),
        software_version: z.string(),
        bundle_digest: hash,
        contract_sha256: hash,
        scope: z.string(),
      })
      .strict(),
  ),
};
export const manifestSchema = z
  .object({
    schema_version: z.literal(VERSION),
    hash_mode: z.literal("canonical-lf-utf8"),
    files: z
      .array(
        z
          .object({
            path: z.string(),
            sha256: hash,
            bytes: count,
          })
          .strict(),
      )
      .length(Object.keys(schemas).length),
  })
  .strict();
export type Evidence = {
  [K in keyof typeof schemas]: z.infer<(typeof schemas)[K]>;
};
export type Manifest = z.infer<typeof manifestSchema>;
export type Pathway = z.infer<typeof pathway>;
