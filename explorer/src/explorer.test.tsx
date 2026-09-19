import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { render, screen, fireEvent, within, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import ts from "typescript";
import { loadEvidence, canonical, sha256 } from "./data/load";
import { schemas } from "./types/evidence";
import { routes } from "./lib/routes";
import { Explorer } from "./App";
import { HashValue } from "./components/HashValue";

const get = async (name: string) =>
  readFileSync(resolve("public/evidence", name), "utf8");
let bundle: Awaited<ReturnType<typeof loadEvidence>>;
beforeAll(async () => {
  bundle = await loadEvidence(get);
  // A cold install may spend seconds transforming Recharts. Load the real module
  // as fixture setup; UI assertions must not race that dependency compilation.
  await import("./charts/Sensitivity");
}, 30000);
describe("distributed evidence integrity", () => {
  it("loads every expected file with schema and hash validation", () => {
    expect(Object.keys(bundle.evidence).sort()).toEqual(
      Object.keys(schemas).sort(),
    );
    expect(
      bundle.evidence.rdx.data.pathways.map((p) => [
        p.numerator,
        p.denominator,
      ]),
    ).toEqual([
      [22539, 32133],
      [9, 7776],
      [4, 130],
      [2, 65],
    ]);
  });
  it("fails closed on corrupt projections", async () => {
    await expect(
      loadEvidence(async (p) => (await get(p)) + (p === "rdx.json" ? " " : "")),
    ).rejects.toThrow("integrity mismatch");
  });
  it("rejects an unexpected roster before reading arbitrary paths", async () => {
    await expect(
      loadEvidence(async (p) => {
        const text = await get(p);
        return p === "manifest.json"
          ? text.replace("rdx.json", "../../runs/private.json")
          : text;
      }),
    ).rejects.toThrow("roster");
  });
  it("also rejects a malformed schema with a matching content hash", async () => {
    const malformed = JSON.parse(await get("rdx.json"));
    malformed.data.unexpected = 1;
    const text = JSON.stringify(malformed);
    const manifest = JSON.parse(await get("manifest.json"));
    Object.assign(
      manifest.files.find((f: { path: string }) => f.path === "rdx.json"),
      { bytes: canonical(text).length, sha256: await sha256(canonical(text)) },
    );
    await expect(
      loadEvidence(async (p) =>
        p === "manifest.json"
          ? JSON.stringify(manifest)
          : p === "rdx.json"
            ? text
            : get(p),
      ),
    ).rejects.toThrow();
  });
  it("is cross-platform canonical", async () => {
    expect(
      await loadEvidence(async (p) => (await get(p)).replaceAll("\n", "\r\n")),
    ).toEqual(bundle);
  });
});
describe("five evidence routes", () => {
  it("initializes React imports before creating the lazy recoverability chart", () => {
    // Vite's dev CommonJS interop creates const bindings at the import location.
    // A late React import can pass build/render tests yet throw a browser TDZ.
    const source = ts.createSourceFile(
      "Recoverability.tsx",
      readFileSync(resolve("src/pages/Recoverability.tsx"), "utf8"),
      ts.ScriptTarget.Latest,
      true,
      ts.ScriptKind.TSX,
    );
    const firstRuntimeStatement = source.statements.findIndex(
      (statement) => !ts.isImportDeclaration(statement),
    );
    const reactImport = source.statements.findIndex(
      (statement) =>
        ts.isImportDeclaration(statement) &&
        ts.isStringLiteral(statement.moduleSpecifier) &&
        statement.moduleSpecifier.text === "react",
    );
    expect(reactImport).toBeGreaterThanOrEqual(0);
    expect(reactImport).toBeLessThan(firstRuntimeStatement);
    expect(
      source.statements.slice(firstRuntimeStatement).some(ts.isImportDeclaration),
    ).toBe(false);
  });
  for (const route of routes)
    it(`renders ${route.id} without a backend`, () => {
      window.history.replaceState(null, "", `#/${route.id}`);
      render(<Explorer {...bundle} />);
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
      expect(
        screen
          .getByRole("link", { name: new RegExp(route.label) })
          .getAttribute("aria-current"),
      ).toBe("page");
    });
  it("inspects transfer cells and switches published metrics", () => {
    window.history.replaceState(null, "", "#/transfer");
    render(<Explorer {...bundle} />);
    const cell = screen.getByRole("button", { name: /B to T:/ });
    fireEvent.click(cell);
    expect(cell.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText("Trained on B")).toBeTruthy();
    expect(screen.getByText("Evaluated on T")).toBeTruthy();
    expect(screen.getByText("FPR budget ratio", { selector: "dt" })).toBeTruthy();
    expect(screen.getByText("Mean FPR divided by the frozen 0.001 target.")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Metric"), {
      target: { value: "fpr_budget_ratio" },
    });
    expect(
      screen.getByRole("button", {
        name: /B to T: Mean false-positive budget ratio 13.6/,
      }),
    ).toBeTruthy();
    expect(screen.getByText(/not underlying exact scores/)).toBeTruthy();
    expect(screen.getByText(/Rounded published F03 annotations/)).toBeTruthy();
  });
  it("preserves censoring, timing and zero medians", async () => {
    window.history.replaceState(null, "", "#/recoverability");
    render(<Explorer {...bundle} />);
    expect(screen.getByText(/censored—not failures/)).toBeTruthy();
    await screen.findByLabelText("Capability estimand");
    expect(screen.getByText("historical")).toBeTruthy();
    expect(screen.getAllByText("prospective")).toHaveLength(2);
    expect(screen.getAllByText(/Median 0 pp/)).toHaveLength(3);
    expect(screen.getByText(/not Core-selectable/)).toBeTruthy();
    for (const tag of ["SAME-WINDOW", "OFFLINE ORACLE", "DEPLOYED", "DEPLOYED · ASSESSABLE ONLY"])
      expect(screen.getByText(tag, { exact: true })).toBeTruthy();
    const gate = within(screen.getByRole("region", { name: "Frozen follow-up gate" }));
    expect(gate.getByText("GATE NOT MET")).toBeTruthy();
    expect(gate.getByText(/median paired P2 improvement ≥ \+2.0 pp/)).toBeTruthy();
    expect(gate.getByText(/positive rotation median in ≥ 3\/4 rotations/)).toBeTruthy();
    expect(gate.getAllByText(/median improvement 0.0 pp · positive rotations 1\/4/)).toHaveLength(3);
    expect(gate.getByText(/not a significance test/)).toBeTruthy();
    expect(screen.getByText(bundle.evidence.rdx.data.interpretation)).toBeTruthy();
    fireEvent.click(screen.getByText("Exact published values and estimand definitions"));
    expect(screen.getByRole("region", { name: /Exact capability values/ }).getAttribute("tabindex")).toBe("0");
    fireEvent.change(screen.getByLabelText("Capability estimand"), {
      target: { value: "p1" },
    });
    expect(
      screen.getByRole("img", { name: /P1 mean and median/ }),
    ).toBeTruthy();
  });
  it("shows the complete label-free contract and no health calculator", () => {
    window.history.replaceState(null, "", "#/health");
    render(<Explorer {...bundle} />);
    expect(bundle.evidence.model_health.data.features).toHaveLength(28);
    expect(screen.getByText(/never full-window labels/)).toBeTruthy();
    expect(screen.queryAllByRole("spinbutton")).toHaveLength(0);
    expect(screen.getByText("0.001", { exact: true })).toBeTruthy();
    expect(screen.getByText("0.10", { exact: true })).toBeTruthy();
    expect(screen.getByText("0.1%", { exact: true })).toBeTruthy();
    expect(screen.getByText("10 percentage points", { exact: true })).toBeTruthy();
  });
  it("links frozen tags without changing the provenance trust boundary", () => {
    window.history.replaceState(null, "", "#/provenance");
    render(<Explorer {...bundle} />);
    const p = bundle.evidence.provenance.data;
    for (const tag of [p.freeze_tag, p.rdx_tag])
      expect(screen.getByRole("link", { name: tag }).getAttribute("href")).toBe(
        `https://github.com/GavinSaiun/DANIDS-Research/tree/${encodeURIComponent(tag)}`,
      );
    expect(screen.getByRole("button", { name: "Copy analysis bundle digest" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Copy analysis-contract SHA-256" })).toBeTruthy();
    expect(screen.getByText(/not malicious replacement of both/)).toBeTruthy();
    expect(screen.getByText(/no remote archive or tag verification/)).toBeTruthy();
  });
});

describe("presentation safeguards", () => {
  it("disables the Recharts categorical cursor while keeping tooltip formatting", () => {
    const source = readFileSync(resolve("src/charts/Sensitivity.tsx"), "utf8");
    expect(source).toMatch(/<Tooltip\s+cursor=\{false\}/);
    expect(source).toContain("contentStyle=");
    expect(source).toContain("formatter=");
  });
  it("provides a local favicon compatible with a deployment base path", () => {
    expect(readFileSync(resolve("index.html"), "utf8")).toContain('href="%BASE_URL%favicon.svg"');
    expect(readFileSync(resolve("public/favicon.svg"), "utf8")).toContain('<svg xmlns="http://www.w3.org/2000/svg"');
  });
  it("copies an exact SHA-256 with accessible feedback", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const original = Object.getOwnPropertyDescriptor(navigator, "clipboard");
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    try {
      const value = bundle.evidence.provenance.data.bundle_digest;
      render(<HashValue value={value} label="test SHA-256" />);
      fireEvent.click(screen.getByRole("button", { name: "Copy test SHA-256" }));
      await waitFor(() => expect(screen.getByRole("status").textContent).toBe("Copied."));
      expect(writeText).toHaveBeenCalledExactlyOnceWith(value);
    } finally {
      if (original) Object.defineProperty(navigator, "clipboard", original);
      else Reflect.deleteProperty(navigator, "clipboard");
    }
  });
  it.each([undefined, { writeText: vi.fn().mockRejectedValue(new Error("Denied")) }])(
    "keeps hashes selectable when clipboard access is unavailable (%s)",
    async (clipboard) => {
      const original = Object.getOwnPropertyDescriptor(navigator, "clipboard");
      Object.defineProperty(navigator, "clipboard", { configurable: true, value: clipboard });
      try {
        const value = bundle.evidence.provenance.data.contract_sha256;
        render(<HashValue value={value} label="test SHA-256" />);
        fireEvent.click(screen.getByRole("button", { name: "Copy test SHA-256" }));
        await waitFor(() => expect(screen.getByRole("status").textContent).toBe("Copy unavailable. Select and copy the hash above."));
        expect(screen.getByText(value)).toBeTruthy();
      } finally {
        if (original) Object.defineProperty(navigator, "clipboard", original);
        else Reflect.deleteProperty(navigator, "clipboard");
      }
    },
  );
});
