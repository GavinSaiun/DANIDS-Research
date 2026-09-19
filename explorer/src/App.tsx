import { useEffect, useRef, useState } from "react";
import { loadEvidence } from "./data/load";
import type { Evidence, Manifest } from "./types/evidence";
import { currentPage, routes } from "./lib/routes";
import { Overview } from "./pages/Overview";
import { Transfer } from "./pages/Transfer";
import { Health } from "./pages/Health";
import { Recoverability } from "./pages/Recoverability";
import { Provenance } from "./pages/Provenance";

export function Explorer({
  evidence,
  manifest,
}: {
  evidence: Evidence;
  manifest: Manifest;
}) {
  const [page, setPage] = useState(currentPage);
  const main = useRef<HTMLElement>(null);
  useEffect(() => {
    const navigate = () => {
      setPage(currentPage());
      main.current?.focus();
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  useEffect(() => {
    document.title = `${routes.find((r) => r.id === page)?.label ?? "Not found"} · DANIDS Explorer`;
  }, [page]);
  return (
    <>
      <a
        className="skip-link"
        href="#content"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        Skip to evidence
      </a>
      <div className="app-shell">
        <aside className="sidebar">
          <a href="#/overview" className="brand">
            <span className="brand-mark" aria-hidden="true">
              D
            </span>
            <span>
              DANIDS<small>EXPLORER</small>
            </span>
          </a>
          <p className="sidebar-label">Research, made inspectable.</p>
          <nav aria-label="Evidence pages">
            {routes.map((r) => (
              <a
                key={r.id}
                href={`#/${r.id}`}
                aria-current={page === r.id ? "page" : undefined}
              >
                <span>{r.number}</span>
                {r.label}
              </a>
            ))}
          </nav>
          <div className="sidebar-footer">
            <span className="read-only-dot" />
            Read-only evidence
            <br />
            <small>Studies 1–5 + post-freeze RDX</small>
          </div>
        </aside>
        <div className="content-shell">
          <div className="topbar">
            <span>DEPLOYMENT-AWARE NETWORK INTRUSION DETECTION</span>
            <span>Public evidence · v1</span>
          </div>
          <main id="content" tabIndex={-1} ref={main}>
            {page === "overview" && <Overview evidence={evidence} />}
            {page === "transfer" && (
              <Transfer data={evidence.cross_domain.data} />
            )}
            {page === "health" && <Health data={evidence.model_health.data} />}
            {page === "recoverability" && (
              <Recoverability data={evidence.rdx.data} />
            )}
            {page === "provenance" && (
              <Provenance evidence={evidence} manifest={manifest} />
            )}
            {page === "not-found" && (
              <>
                <h1>Page not found</h1>
                <a href="#/overview">Return to overview</a>
              </>
            )}
          </main>
          <footer>
            Frozen evidence. Bounded claims. No experiments run here.
            <a href="#/provenance">Inspect provenance ↗</a>
          </footer>
        </div>
      </div>
    </>
  );
}
export default function App() {
  const [bundle, setBundle] = useState<{
    evidence: Evidence;
    manifest: Manifest;
  }>();
  const [error, setError] = useState<string>();
  useEffect(() => {
    let active = true;
    loadEvidence()
      .then((b) => {
        if (active) setBundle(b);
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, []);
  if (error)
    return (
      <main className="loading" role="alert">
        <p className="eyebrow">Evidence verification failed</p>
        <h1>No unverified results will be displayed.</h1>
        <p>{error}</p>
        <p>
          Check the distributed evidence files and manifest. There is no
          fallback dataset.
        </p>
      </main>
    );
  if (!bundle)
    return (
      <main className="loading" role="status">
        <p className="eyebrow">DANIDS Explorer</p>
        <h1>Verifying the evidence…</h1>
        <p>Checking source projections before rendering.</p>
      </main>
    );
  return <Explorer {...bundle} />;
}
