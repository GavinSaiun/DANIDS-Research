import { lazy, Suspense } from "react";
import type { Evidence } from "../types/evidence";
import {
  Detail,
  Heading,
  Note,
  PathwayCard,
  SourceNote,
} from "../components/Primitives";
import { number } from "../lib/routes";

const Sensitivity = lazy(() =>
  import("../charts/Sensitivity").then((m) => ({ default: m.Sensitivity })),
);

export function Recoverability({ data }: { data: Evidence["rdx"]["data"] }) {
  return (
    <>
      <Heading
        eyebrow="Post-freeze RDX / Recoverability"
        title="Where recognition stops short of repair."
      >
        Three distinct questions: was harm recognised, was a successful
        intervention available, and did a deployed intervention recover?
      </Heading>
      <div className="pathways">
        {data.pathways.map((p) => (
          <PathwayCard key={p.id} item={p} showProvenance />
        ))}
      </div>
      <Note>
        Only assessable two-window horizons enter sustained recovery.
        Unassessable interventions remain censored—not failures. These pathways
        are not consecutive stages of one population.
      </Note>
      <div className="two-up">
        <section className="panel">
          <h2>Recognition is not capability.</h2>
          <p>
            Core recognised {number(data.core_recognised)} of{" "}
            {number(data.core_harmful)} evaluator-HARMFUL windows. Recognition
            uses the same window, not a future-harm forecast.
          </p>
          <p>
            The Offline Oracle is a non-deployable one-step comparator. It can
            inspect current truth, candidate audit outcomes and the first fresh
            same-domain successor, but not permanent holdouts or the later
            trajectory.
          </p>
        </section>
        <section className="panel">
          <h2>Success has a strict meaning.</h2>
          <p>
            A feasible action must execute successfully, pass the candidate
            audit, and yield a SAFE first fresh same-domain successor. UNCERTAIN
            is not confirmed success.
          </p>
          <p>
            Candidate execution is isolated; audit-gated promotion and exact
            rollback preserve the deployed trajectory. Failed repair is bounded
            to the frozen B100/D1 and A0–A4 regime.
          </p>
        </section>
      </div>
      <section>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Frozen intervention family</p>
            <h2>Five actions. No extra evidence.</h2>
          </div>
        </div>
        <div className="actions">
          {data.actions.map((a) => (
            <article key={a.id}>
              <span>{a.id}</span>
              <h3>{a.name.replaceAll("_", " ")}</h3>
              <p>{a.detail}</p>
            </article>
          ))}
        </div>
      </section>
      <Detail title="Label, replay and audit constraints">
        <p>
          The original Study-4 regime uses B100, query size 25, one-window
          delay, and 20 train / 5 audit rows per release. Historical replay and
          audit capacities are 400 / 100 per domain. Permanent holdouts never
          influence querying, training or Oracle action choice.
        </p>
        <p>
          The RDX sensitivity varies only optimizer-eligible target evidence.
          Audit, optimizer, threshold, replay, action and delay rules remain
          frozen.
        </p>
      </Detail>
      <Suspense
        fallback={<p role="status">Loading training-evidence display…</p>}
      >
        <Sensitivity data={data} />
      </Suspense>
      <SourceNote paths={["RDX_RESULTS.md", "docs/decisions.md"]} />
    </>
  );
}
