import type { Evidence } from "../types/evidence";
import {
  Heading,
  Note,
  PathwayCard,
  SourceNote,
} from "../components/Primitives";
import { number } from "../lib/routes";

export function Overview({ evidence: e }: { evidence: Evidence }) {
  const core = e.study4.data.methods.DANIDS_CORE;
  const always = e.study4.data.methods.ALWAYS_ADAPT;
  return (
    <div className="overview">
      <Heading
        eyebrow="DANIDS Explorer / Frozen evidence"
        title="Recognising harm is not the same as repairing it."
      >
        A read-only guide to sequential network-domain shift, model health and
        constrained recoverability. Studies 1–5 and the separately frozen RDX
        extension.
      </Heading>
      <ol className="chain">
        {e.overview.data.chain.map((step, i) => (
          <li key={step}>
            <span>{String(i + 1).padStart(2, "0")}</span>
            {step}
          </li>
        ))}
      </ol>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Recognition → capability → deployment</p>
          <h2>Different questions. Different denominators.</h2>
        </div>
        <a className="text-link" href="#/recoverability">
          Explore recoverability ↗
        </a>
      </div>
      <div className="pathways">
        {e.rdx.data.pathways.map((p) => (
          <PathwayCard key={p.id} item={p} />
        ))}
      </div>
      <Note>
        These are separate populations, not a nested funnel. Censored recovery
        horizons are not failures. Counts are nested evidence, not independent
        experimental replicates.
      </Note>
      <div className="two-up">
        <section className="panel">
          <p className="eyebrow">Frozen Study 4</p>
          <h2>
            Sparser interventions.
            <br />
            Not sparser supervision.
          </h2>
          <div className="resource-pair">
            <div>
              <strong>{core.accepted_updates}</strong>
              <span>Core accepted updates</span>
            </div>
            <div>
              <strong>{always.accepted_updates}</strong>
              <span>Always-Adapt accepted updates</span>
            </div>
          </div>
          <p>
            Both requested {number(core.labels_requested)} labels. Core reduced
            accepted update frequency but did not achieve comparable safety
            under the frozen B100/D1 and A0–A4 regime.
          </p>
        </section>
        <section className="panel">
          <p className="eyebrow">Interpretation boundary</p>
          <h2>What the evidence does—and does not—say.</h2>
          <ul className="spaced-list">
            {e.overview.data.limitations.map((x) => (
              <li key={x}>{x}</li>
            ))}
          </ul>
        </section>
      </div>
      <SourceNote
        paths={[
          "RDX_RESULTS.md",
          "docs/thesis_evidence_freeze.md",
          "thesis/assets/visual_manifest.json",
        ]}
      />
    </div>
  );
}
