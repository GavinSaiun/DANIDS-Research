import type { ReactNode } from "react";
import type { Pathway } from "../types/evidence";
import { number } from "../lib/routes";

export function Heading({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <header className="page-heading">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="lede">{children}</p>
    </header>
  );
}
export function Note({ children }: { children: ReactNode }) {
  return <aside className="note">{children}</aside>;
}
export function Detail({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <details>
      <summary>{title}</summary>
      <div className="details-content">{children}</div>
    </details>
  );
}
const pathwayProvenance: Record<Pathway["id"], string> = {
  recognition: "SAME-WINDOW",
  capability: "OFFLINE ORACLE",
  immediate: "DEPLOYED",
  sustained: "DEPLOYED · ASSESSABLE ONLY",
};

export function PathwayCard({
  item,
  showProvenance = false,
}: {
  item: Pathway;
  showProvenance?: boolean;
}) {
  return (
    <article className={`pathway ${item.id}`}>
      <p className="eyebrow">{item.title}</p>
      {showProvenance && (
        <span className="provenance-tag">{pathwayProvenance[item.id]}</span>
      )}
      <div className="stat">{item.percent}</div>
      <p>{item.outcome}</p>
      <div className="ratio">
        <strong>{number(item.numerator)}</strong>
        <span>of {number(item.denominator)}</span>
      </div>
      <p className="unit">Denominator: {item.unit}</p>
      <p className="muted">{item.population}</p>
    </article>
  );
}
export function SourceNote({ paths }: { paths: string[] }) {
  return (
    <p className="source-note">
      Public source: {paths.join(" · ")}. Source hashes in Evidence &
      provenance.
    </p>
  );
}
