import type { Evidence, Manifest } from "../types/evidence";
import { Detail, Heading, Note } from "../components/Primitives";
import { HashValue } from "../components/HashValue";

const tagUrl = (tag: string) =>
  `https://github.com/GavinSaiun/DANIDS-Research/tree/${encodeURIComponent(tag)}`;

export function Provenance({
  evidence,
  manifest,
}: {
  evidence: Evidence;
  manifest: Manifest;
}) {
  const p = evidence.provenance.data;
  const sources = [
    ...new Map(
      Object.values(evidence)
        .flatMap((e) => e.sources)
        .map((s) => [s.path, s]),
    ).values(),
  ];
  return (
    <>
      <Heading
        eyebrow="Evidence / Provenance"
        title="Every display has a paper trail."
      >
        Frozen scientific evidence, post-freeze diagnostics, software releases
        and public projections are separate identities.
      </Heading>
      <div className="three-up">
        <article className="panel">
          <p className="eyebrow">Studies 1–5</p>
          <h2>DANIDS 2.0</h2>
          <a href={tagUrl(p.freeze_tag)}><code>{p.freeze_tag}</code></a>
          <p>
            The original thesis evidence freeze. Hypothesis verdicts remain
            unchanged.
          </p>
        </article>
        <article className="panel">
          <p className="eyebrow">Post-freeze diagnostics</p>
          <h2>RDX</h2>
          <a href={tagUrl(p.rdx_tag)}><code>{p.rdx_tag}</code></a>
          <p>
            Separate protocols and mixed historical/prospective sensitivity
            evidence. Not Study 6.
          </p>
        </article>
        <article className="panel">
          <p className="eyebrow">Research software</p>
          <h2>{p.software_version}</h2>
          <p>A software version, not a new scientific freeze.</p>
        </article>
      </div>
      <section className="panel">
        <h2>Frozen RDX identities</h2>
        <dl className="hashes">
          <dt>Analysis bundle digest</dt>
          <dd>
            <HashValue value={p.bundle_digest} label="analysis bundle digest" />
          </dd>
          <dt>Analysis-contract SHA-256</dt>
          <dd>
            <HashValue value={p.contract_sha256} label="analysis-contract SHA-256" />
          </dd>
        </dl>
        <p>{p.scope}</p>
      </section>
      <Note>
        The browser verifies canonical JSON against the distributed manifest.
        This detects projection corruption, not malicious replacement of both
        files and manifest. Reviewed source pins and repository history supply
        the trust boundary; no remote archive or tag verification is performed
        in the browser.
      </Note>
      <Detail title="Frozen hypothesis ledger">
        <div className="ledger">
          {Object.entries(evidence.overview.data.ledger).map(([h, verdict]) => (
            <div key={h}>
              <strong>{h}</strong>
              <span>{verdict}</span>
            </div>
          ))}
        </div>
        <p>
          Copied from the evidence freeze, never recomputed. Study 6: NO-GO.
        </p>
      </Detail>
      <section className="panel">
        <h2>Public sources</h2>
        <p>
          UTF-8 text with CRLF/CR canonicalised to LF. No ignored data or
          experiment outputs are requested by the browser.
        </p>
        {sources.map((s) => (
          <details key={s.path}>
            <summary>{s.path}</summary>
            <HashValue value={s.sha256} label={`${s.path} source SHA-256`} />
          </details>
        ))}
      </section>
      <section className="panel">
        <h2>Projection manifest</h2>
        <p>
          All files below passed schema, byte-count and SHA-256 checks before
          this page was rendered.
        </p>
        {manifest.files.map((f) => (
          <details key={f.path}>
            <summary>
              {f.path} · {f.bytes.toLocaleString("en-US")} canonical bytes
            </summary>
            <HashValue value={f.sha256} label={`${f.path} projection SHA-256`} />
            <p>
              <a
                href={`${import.meta.env.BASE_URL}evidence/${f.path}`}
                download
              >
                Download projection JSON
              </a>
            </p>
          </details>
        ))}
        <a href={`${import.meta.env.BASE_URL}evidence/manifest.json`} download>
          Download manifest
        </a>
      </section>
    </>
  );
}
