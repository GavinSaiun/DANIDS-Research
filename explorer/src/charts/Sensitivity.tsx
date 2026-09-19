import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Evidence } from "../types/evidence";
import { Detail } from "../components/Primitives";
import { number } from "../lib/routes";

export function Sensitivity({ data }: { data: Evidence["rdx"]["data"] }) {
  const [estimand, setEstimand] = useState<"p1" | "p2">("p2");
  return (
    <section className="panel sensitivity">
      <div className="section-heading">
        <div>
          <p className="eyebrow">
            Training-evidence sensitivity · post-freeze RDX
          </p>
          <h2>More training evidence. A bounded change.</h2>
        </div>
        <label>
          Capability estimand
          <select
            value={estimand}
            onChange={(e) => setEstimand(e.target.value as "p1" | "p2")}
          >
            <option value="p2">P2 · A1 / A2 / A4</option>
            <option value="p1">P1 · A0–A4</option>
          </select>
        </label>
      </div>
      <p>
        Mean and median capability over {data.paired_units} rotation × seed
        units. Unit: percent of eligible currently-HARMFUL Oracle decisions
        within each unit.
      </p>
      <div
        className="chart"
        role="img"
        aria-label={`${estimand.toUpperCase()} mean and median capability by budget; exact values available below`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data.budgets}
            margin={{ top: 12, right: 24, bottom: 8, left: 0 }}
            accessibilityLayer
          >
            <CartesianGrid stroke="#2a3a48" vertical={false} />
            <XAxis dataKey="budget" stroke="#becbd5" tickLine={false} />
            <YAxis stroke="#becbd5" tickFormatter={(v) => `${v}%`} width={65} />
            <Tooltip
              cursor={false}
              contentStyle={{
                background: "#14212c",
                borderColor: "#698393",
                color: "#f0f5f7",
              }}
              formatter={(v) => `${v}%`}
            />
            <Legend />
            <Bar
              dataKey={`mean_${estimand}`}
              name="Mean (%)"
              fill="#71d6c6"
              radius={[4, 4, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              dataKey={`median_${estimand}`}
              name="Median (%) · all zero"
              fill="#f1c77b"
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="budget-labels">
        {data.budgets.map((b) => (
          <div key={b.budget}>
            <strong>{b.budget}</strong>
            <span>{b.timing}</span>
            <span>{number(b.optimizer_rows)} optimizer rows</span>
            <span>{number(b.eligible)} eligible decisions</span>
          </div>
        ))}
      </div>
      <Detail title="Exact published values and estimand definitions">
        <p>
          P1: any confirmed A0–A4 success. P2: any confirmed A1/A2/A4 success.
          These are complete-counterfactual capability rates, not
          selected-action success rates. The pooled capability percentage
          elsewhere is not this unweighted mean.
        </p>
        <div
          className="table-scroll"
          role="region"
          aria-label="Exact capability values (scroll horizontally if needed)"
          tabIndex={0}
        >
          <table>
            <caption>
              Published percentages, summarized per rotation × seed
            </caption>
            <thead>
              <tr>
                <th>Budget</th>
                <th>Mean P1</th>
                <th>Median P1</th>
                <th>Mean P2</th>
                <th>Median P2</th>
              </tr>
            </thead>
            <tbody>
              {data.budgets.map((b) => (
                <tr key={b.budget}>
                  <th>{b.budget}</th>
                  <td>{b.mean_p1.toFixed(3)}%</td>
                  <td>{b.median_p1}%</td>
                  <td>{b.mean_p2.toFixed(3)}%</td>
                  <td>{b.median_p2}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Detail>
      <h3>Paired P2 contrasts</h3>
      <div className="contrast-grid">
        {data.contrasts.map((c) => (
          <article key={c.contrast}>
            <h4>{c.contrast}</h4>
            <strong>+{c.mean_pp.toFixed(3)} pp</strong>
            <p>Mean paired difference</p>
            <p>
              Median {c.median_pp} pp · positive rotations{" "}
              {c.positive_rotations}
            </p>
          </article>
        ))}
      </div>
      <section className="gate" aria-label="Frozen follow-up gate">
        <h3>Frozen follow-up gate</h3>
        <strong className="gate-result">GATE NOT MET</strong>
        <p>
          Required: median paired P2 improvement ≥ +
          {data.gate_median_pp.toFixed(1)} pp and positive rotation median
          in ≥ {data.gate_positive_rotations} rotations.
        </p>
        <p>Observed paired P2 contrasts:</p>
        <ul className="gate-observed">
          {data.contrasts.map((c) => (
            <li key={c.contrast}>
              <span>{c.contrast}</span>: median improvement{" "}
              {c.median_pp.toFixed(1)} pp · positive rotations{" "}
              {c.positive_rotations}
            </li>
          ))}
        </ul>
        <p>
          A frozen descriptive gate, not a significance test. Windows are not
          independent replicates.
        </p>
      </section>
      <p className="decision">{data.decision}</p>
      <blockquote>{data.interpretation}</blockquote>
      <p className="muted">
        This does not show that more data can never help or that safe adaptation
        is impossible generally.
      </p>
    </section>
  );
}
