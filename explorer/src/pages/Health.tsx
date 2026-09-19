import type { Evidence } from "../types/evidence";
import { Detail, Heading, Note, SourceNote } from "../components/Primitives";

export function Health({ data }: { data: Evidence["model_health"]["data"] }) {
  return (
    <>
      <Heading
        eyebrow="Study 3 / Model health"
        title="Shift is observable. Harm is operational."
      >
        Same-window harm screening connects observable health signals to
        operating-envelope violations. It is not an anticipatory forecast.
      </Heading>
      <div className="envelope">
        <div>
          <span>Target false-positive rate</span>
          <strong>{data.target_fpr}</strong>
          <small className="constant-help">0.1%</small>
        </div>
        <div>
          <span>Recall-loss tolerance</span>
          <strong>{data.recall_loss_tolerance.toFixed(2)}</strong>
          <small className="constant-help">10 percentage points</small>
        </div>
        <div>
          <span>Uncertainty semantics</span>
          <strong>{data.confidence}</strong>
        </div>
      </div>
      <div className="states">
        {data.states.map((s) => (
          <article
            key={s.name}
            className={`panel state ${s.name.toLowerCase()}`}
          >
            <span className="status-label">{s.name}</span>
            <h2>
              {s.name === "SAFE"
                ? "Both conditions hold."
                : s.name === "HARMFUL"
                  ? "Evidence of a violation."
                  : "Evidence is inconclusive."}
            </h2>
            <p>{s.rule}</p>
          </article>
        ))}
      </div>
      <Note>
        These are evaluator-only truth definitions. The deployed controller
        receives a label-free prediction, never full-window labels or evaluator
        SAFE/HARMFUL truth. PREDICTED_SAFE is not certified safety.
      </Note>
      <section className="panel">
        <p className="eyebrow">The information boundary</p>
        <h2>Distribution shift ≠ operational harm</h2>
        <p>
          A shifted distribution does not by itself demonstrate a false-alarm or
          recall violation. Five observable feature groups support the frozen
          health screen.
        </p>
        <ol className="feature-groups">
          {data.feature_groups.map((group, i) => (
            <li key={group}>
              <span>{String(i + 1).padStart(2, "0")}</span>
              {group}
            </li>
          ))}
        </ol>
        <Detail
          title={`Inspect the exact ${data.features.length}-feature contract`}
        >
          <p>
            Ordered combined_unlabelled contract. No complete target labels or
            domain identifiers enter this vector.
          </p>
          <ol className="features">
            {data.features.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ol>
        </Detail>
      </section>
      <SourceNote
        paths={[
          "src/danids/config/core.py",
          "src/danids/health/states.py",
          "docs/decisions.md",
          "docs/thesis_evidence_freeze.md",
        ]}
      />
    </>
  );
}
