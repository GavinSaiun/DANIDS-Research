import { useState, type CSSProperties } from "react";
import type { Evidence } from "../types/evidence";
import { Detail, Heading, Note, SourceNote } from "../components/Primitives";

export function Transfer({ data }: { data: Evidence["cross_domain"]["data"] }) {
  const [metric, setMetric] = useState<"recall" | "fpr_budget_ratio">("recall");
  const [selected, setSelected] = useState(
    data.cells.find((c) => c.source === "U" && c.target === "T")!,
  );
  const label =
    metric === "recall"
      ? "Mean attack recall / TPR"
      : "Mean false-positive budget ratio";
  return (
    <>
      <Heading
        eyebrow="Study 1 / Static transfer"
        title="A detector does not travel unchanged."
      >
        Transfer is directional and target-dependent. Select a cell to inspect
        the published cross-domain result.
      </Heading>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>{label}</h2>
            <p className="muted">
              {data.aggregation}. {data.precision}.
            </p>
          </div>
          <label>
            Metric
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as typeof metric)}
            >
              <option value="recall">Attack recall / TPR</option>
              <option value="fpr_budget_ratio">FPR budget ratio</option>
            </select>
          </label>
        </div>
        <div className="matrix-layout">
          <div className="matrix-wrap">
            <p className="axis-title">Target domain →</p>
            <div className="matrix">
              <span className="axis-corner">Source ↓</span>
              {data.domains.map((d) => (
                <strong key={d} className="col-label">
                  {d}
                </strong>
              ))}
              {data.domains.map((source) => (
                <div className="matrix-row" key={source}>
                  <strong className="row-label">{source}</strong>
                  {data.domains.map((target) => {
                    const cell = data.cells.find(
                      (c) => c.source === source && c.target === target,
                    )!;
                    const value = Number(cell[metric]);
                    const intensity =
                      metric === "recall" ? value : Math.log10(1 + value) / 3;
                    return (
                      <button
                        key={target}
                        className={`matrix-cell ${metric}`}
                        style={{ "--intensity": intensity } as CSSProperties}
                        aria-label={`${source} to ${target}: ${label} ${cell[metric]}`}
                        aria-pressed={
                          selected.source === source &&
                          selected.target === target
                        }
                        onClick={() => setSelected(cell)}
                      >
                        {cell[metric]}
                        {source === target && <span>reference</span>}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
            <p className="muted">
              {metric === "recall"
                ? "Fraction of true attacks detected · darker = lower recall"
                : "FPR divided by the frozen 0.001 budget · logarithmic colour intensity; >1 exceeds the point-estimate budget"}
            </p>
          </div>
          <aside className="cell-inspector" aria-live="polite">
            <p className="eyebrow">Cell inspection</p>
            <h2>
              {selected.source} → {selected.target}
            </h2>
            <p className="transfer-direction">
              <span>Trained on {selected.source}</span>
              <span>Evaluated on {selected.target}</span>
            </p>
            <p>
              {selected.source === selected.target
                ? "Within-domain reference"
                : "Directional cross-domain transfer"}
            </p>
            <dl>
              <dt>Mean attack recall</dt>
              <dd>{selected.recall}</dd>
              <dt>FPR budget ratio</dt>
              <dd>{selected.fpr_budget_ratio} ×</dd>
            </dl>
            <p className="muted">
              Mean FPR divided by the frozen 0.001 target.
            </p>
            <p className="muted">
              Rounded published F03 annotations, not underlying exact scores. Colours
              are presentation aids, not evaluator health labels.
            </p>
          </aside>
        </div>
      </section>
      <Note>
        Source-target reporting positions are not causal deployment-order
        effects. Recall alone cannot establish operational safety.
      </Note>
      <Detail title="Domain key and unavailable metrics">
        <p>
          U: NF-UNSW-NB15-v3 · T: NF-ToN-IoT-v3 · B: NF-BoT-IoT-v3 · C:
          NF-CSE-CIC-IDS2018-v3.
        </p>
        <p>
          Not exposed in this clean-clone projection:{" "}
          {data.unavailable.join(", ")}. False positives per million are
          intentionally not derived from rounded figure values.
        </p>
      </Detail>
      <SourceNote
        paths={[
          "thesis/assets/svg/F03_static_transfer.svg",
          "thesis/assets/visual_manifest.json",
        ]}
      />
    </>
  );
}
